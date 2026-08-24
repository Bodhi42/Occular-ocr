//! Чтение `compact_lm.npz` напрямую из Rust — без материализации массивов в Python.
//!
//! `np.savez` пишет zip без сжатия (STORED), поэтому файл достаточно один раз отобразить в
//! память: массивы n-грамм читаются прямо из отображения. Это убирает и время загрузки
//! (268 МБ больше никто не копирует), и резидентную память (страницы файловые, общие между
//! процессами и вытесняемые под давлением).
//!
//! Разбирается ровно столько формата, сколько нужно: центральный каталог zip (включая zip64,
//! который numpy включает принудительно) и заголовок .npy версий 1.x/2.x.

use std::fs::File;
use std::sync::Arc;

use memmap2::Mmap;

pub struct Npz {
    map: Arc<Mmap>,
    entries: Vec<(String, usize, usize)>, // имя, смещение данных, длина данных в байтах
}

/// Колонка чисел внутри отображения: смещение в байтах + число элементов.
/// Чтения невыровненные — на x86 это тот же одиночный load, а гарантий выравнивания
/// внутри zip-архива нет.
#[derive(Clone)]
pub struct Col {
    map: Arc<Mmap>,
    off: usize,
    len: usize,
}

impl Col {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn u64_at(&self, i: usize) -> u64 {
        let p = self.off + i * 8;
        let b: [u8; 8] = self.map[p..p + 8].try_into().unwrap();
        u64::from_le_bytes(b)
    }

    #[inline(always)]
    pub fn f32_at(&self, i: usize) -> f32 {
        let p = self.off + i * 4;
        let b: [u8; 4] = self.map[p..p + 4].try_into().unwrap();
        f32::from_le_bytes(b)
    }
}

fn rd_u16(b: &[u8], i: usize) -> usize {
    u16::from_le_bytes([b[i], b[i + 1]]) as usize
}

fn rd_u32(b: &[u8], i: usize) -> usize {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize
}

fn rd_u64(b: &[u8], i: usize) -> usize {
    u64::from_le_bytes(b[i..i + 8].try_into().unwrap()) as usize
}

impl Npz {
    pub fn open(path: &str) -> Result<Npz, String> {
        let file = File::open(path).map_err(|e| format!("не открыть {path}: {e}"))?;
        // SAFETY: файл весов не переписывается на ходу; при подмене получим мусор, но не UB
        // в смысле памяти процесса — как и любая другая работа с mmap файлов.
        let map = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap {path}: {e}"))?;
        let buf: &[u8] = &map;

        // ── конец центрального каталога
        let eocd = {
            let n = buf.len();
            let from = n.saturating_sub(66_000);
            (from..n.saturating_sub(21))
                .rev()
                .find(|&i| &buf[i..i + 4] == b"PK\x05\x06")
                .ok_or("не найден конец центрального каталога zip")?
        };
        let mut n_entries = rd_u16(buf, eocd + 10);
        let mut cd_off = rd_u32(buf, eocd + 16);
        if n_entries == 0xFFFF || cd_off == 0xFFFF_FFFF {
            // zip64: локатор лежит прямо перед EOCD
            let loc = eocd.checked_sub(20).ok_or("нет локатора zip64")?;
            if &buf[loc..loc + 4] != b"PK\x06\x07" {
                return Err("ожидался локатор zip64".into());
            }
            let z64 = rd_u64(buf, loc + 8);
            if &buf[z64..z64 + 4] != b"PK\x06\x06" {
                return Err("ожидалась запись zip64".into());
            }
            n_entries = rd_u64(buf, z64 + 32);
            cd_off = rd_u64(buf, z64 + 48);
        }

        // ── записи каталога
        let mut entries = Vec::with_capacity(n_entries);
        let mut p = cd_off;
        for _ in 0..n_entries {
            if &buf[p..p + 4] != b"PK\x01\x02" {
                return Err("повреждён центральный каталог".into());
            }
            let method = rd_u16(buf, p + 10);
            let mut comp_size = rd_u32(buf, p + 20);
            let name_len = rd_u16(buf, p + 28);
            let extra_len = rd_u16(buf, p + 30);
            let comment_len = rd_u16(buf, p + 32);
            let mut local_off = rd_u32(buf, p + 42);
            let name = String::from_utf8_lossy(&buf[p + 46..p + 46 + name_len]).into_owned();

            // zip64-поля вынесены в extra (numpy пишет с force_zip64)
            if comp_size == 0xFFFF_FFFF || local_off == 0xFFFF_FFFF {
                let ex_start = p + 46 + name_len;
                let mut q = ex_start;
                let ex_end = ex_start + extra_len;
                while q + 4 <= ex_end {
                    let id = rd_u16(buf, q);
                    let sz = rd_u16(buf, q + 2);
                    if id == 0x0001 {
                        let mut r = q + 4;
                        if rd_u32(buf, p + 24) == 0xFFFF_FFFF {
                            r += 8; // несжатый размер
                        }
                        if comp_size == 0xFFFF_FFFF {
                            comp_size = rd_u64(buf, r);
                            r += 8;
                        }
                        if local_off == 0xFFFF_FFFF {
                            local_off = rd_u64(buf, r);
                        }
                        break;
                    }
                    q += 4 + sz;
                }
            }
            if method != 0 {
                return Err(format!("{name}: архив со сжатием, нужен npz без компрессии"));
            }

            // данные лежат после локального заголовка (его длины полей свои)
            if &buf[local_off..local_off + 4] != b"PK\x03\x04" {
                return Err(format!("{name}: повреждён локальный заголовок"));
            }
            let l_name = rd_u16(buf, local_off + 26);
            let l_extra = rd_u16(buf, local_off + 28);
            let data_off = local_off + 30 + l_name + l_extra;
            entries.push((name, data_off, comp_size));
            p += 46 + name_len + extra_len + comment_len;
        }

        Ok(Npz { map: Arc::new(map), entries })
    }

    /// Колонка по имени массива (`"k1"` → запись `k1.npy`) с проверкой типа элемента.
    pub fn col(&self, name: &str, dtype: &str) -> Result<Col, String> {
        let want = format!("{name}.npy");
        let (_, off, len) = self
            .entries
            .iter()
            .find(|(n, _, _)| *n == want)
            .ok_or_else(|| format!("в архиве нет {want}"))?;
        let buf: &[u8] = &self.map;
        if &buf[*off..*off + 6] != b"\x93NUMPY" {
            return Err(format!("{want}: это не .npy"));
        }
        let major = buf[*off + 6];
        let (hdr_len, hdr_at) = if major == 1 {
            (rd_u16(buf, *off + 8), *off + 10)
        } else {
            (rd_u32(buf, *off + 8), *off + 12)
        };
        let header = String::from_utf8_lossy(&buf[hdr_at..hdr_at + hdr_len]).into_owned();
        if !header.contains(&format!("'{dtype}'")) {
            return Err(format!("{want}: ожидался dtype {dtype}, заголовок: {header}"));
        }
        if header.contains("'fortran_order': True") {
            return Err(format!("{want}: fortran_order не поддерживается"));
        }
        let elem = if dtype.ends_with('8') { 8 } else { 4 };
        let data_off = hdr_at + hdr_len;
        let n_bytes = *len - (data_off - *off);
        Ok(Col { map: self.map.clone(), off: data_off, len: n_bytes / elem })
    }
}

/// Словарь униграмм поверх отображённого `unigrams.txt`: сортируются только смещения строк,
/// сами слова не копируются.
pub struct MappedVocab {
    map: Arc<Mmap>,
    /// (начало, длина) слов, отсортированные по содержимому
    spans: Vec<(u32, u32)>,
}

impl MappedVocab {
    pub fn open(path: &str) -> Result<MappedVocab, String> {
        let file = File::open(path).map_err(|e| format!("не открыть {path}: {e}"))?;
        // SAFETY: см. Npz::open
        let map = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap {path}: {e}"))?;
        let buf: &[u8] = &map;

        let mut spans: Vec<(u32, u32)> = Vec::with_capacity(2_100_000);
        let mut start = 0usize;
        for i in 0..=buf.len() {
            if i == buf.len() || buf[i] == b'\n' {
                let mut end = i;
                while end > start && (buf[end - 1] == b'\r' || buf[end - 1] == b' ') {
                    end -= 1;
                }
                if end > start {
                    spans.push((start as u32, (end - start) as u32));
                }
                start = i + 1;
            }
        }
        let base = buf.as_ptr();
        // сортировка по содержимому без копирования слов
        spans.sort_unstable_by(|a, b| unsafe {
            let sa = std::slice::from_raw_parts(base.add(a.0 as usize), a.1 as usize);
            let sb = std::slice::from_raw_parts(base.add(b.0 as usize), b.1 as usize);
            sa.cmp(sb)
        });
        spans.dedup_by(|a, b| unsafe {
            let sa = std::slice::from_raw_parts(base.add(a.0 as usize), a.1 as usize);
            let sb = std::slice::from_raw_parts(base.add(b.0 as usize), b.1 as usize);
            sa == sb
        });
        spans.shrink_to_fit();
        Ok(MappedVocab { map: Arc::new(map), spans })
    }

    #[inline(always)]
    fn word(&self, i: usize) -> &[u8] {
        let (o, l) = self.spans[i];
        &self.map[o as usize..o as usize + l as usize]
    }

    pub fn len(&self) -> usize {
        self.spans.len()
    }

    pub fn has_prefix(&self, p: &[u8]) -> bool {
        let (mut lo, mut hi) = (0usize, self.spans.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.word(mid) < p {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo < self.spans.len() && self.word(lo).starts_with(p)
    }
}
