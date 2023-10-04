use std::sync::atomic::AtomicU64;

use macroquad::color::Color;
// so a grid is composed of cells?
// this is becuase if we want to write to the grid
// it would be nice if it remembered what it was doing

// simple ass struct, doesn't even have no impl
#[derive(Debug)]
pub struct Cell {
    pub id: u64,
    pub text: String,
    pub bg_color: Option<Color>,
    pub text_color: Option<Color>,
}

static CELL_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub fn cells_allocated() -> u64 {
    CELL_SEQUENCE.load(std::sync::atomic::Ordering::Relaxed)
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            id: next_id(),
            text: String::new(),
            bg_color: None,
            text_color: None,
        }
    }
}

#[inline]
fn next_id() -> u64 {
    CELL_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

impl Clone for Cell {
    fn clone(&self) -> Self {
        Self {
            id: next_id(),
            text: self.text.clone(),
            ..*self
        }
    }
}

impl Cell {
    #[inline]
    pub fn set_fg(&mut self, fg: impl Into<Option<Color>>) {
        self.text_color = fg.into();
    }

    #[inline]
    pub fn set_bg(&mut self, bg: impl Into<Option<Color>>) {
        self.bg_color = bg.into();
    }

    #[inline]
    pub fn set_text<T>(&mut self, text: impl Into<Option<T>>)
    where
        T: std::fmt::Display,
    {
        use std::fmt::Write as _;
        self.text.clear();
        if let Some(value) = text.into() {
            write!(&mut self.text, "{value}").unwrap();
        }
    }

    #[inline]
    pub fn clear_text(&mut self) {
        self.text.clear()
    }

    #[inline]
    pub fn clear(&mut self) {
        let Self {
            id: _,
            text,
            bg_color,
            text_color,
        } = self;
        text.clear();
        *bg_color = None;
        *text_color = None;
    }
}

impl std::fmt::Write for Cell {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.text.write_str(s)
    }
}
