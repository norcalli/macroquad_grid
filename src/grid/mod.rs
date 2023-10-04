use std::{
    borrow::Cow,
    cell::{Ref, RefCell},
    ops::{Index, IndexMut},
};

use macroquad::prelude::*;

pub mod cell;
pub mod position;

pub use position::Position;

/// # the point of this crate!
/// used to represent and draw a grid to the screen
/// heres the repo: https://github.com/TheDinner22/macroquad_grid
///
/// ## construction
/// use the new method or the default method
///
/// ## notes
///
/// only has private feilds so you interface with it via
/// methods (mainly getters and setters)
///
/// ## stuff you can do
///
/// - creating a grid
/// - selecting a cell
/// - changing selected cells color
/// - changing default cell bg color
/// - changing gap color
/// - changing grids postion with Position enum
/// - setting color of a specific cell
/// - writing text to a specific cell
/// - writing text to the selected cell
/// - getting the selected cell's index
/// - drawing the grid
pub struct Grid {
    width: f32,                   // width of the grid in pixels
    height: f32,                  // height of the grid in pixels
    x_offset: position::Position, // for positioning the grid on the screen
    y_offset: position::Position, // for positioning the grid on the screen

    pub auto_resize_text: bool,
    cols: usize,                            // number of cells
    rows: usize,                            // number of cells
    cell_bg_color: macroquad::color::Color, // color of the cells

    pub gap: f32, // space between cells (in pixels)
    pub gap_color: macroquad::color::Color,

    cells: Vec<Vec<cell::Cell>>,

    column_settings: Vec<ColumnSetting>,
    // // Used for quickly calculating the cell position.
    running_totals_before: RefCell<Vec<f32>>,
    fixed_width_total: f32,
    fixed_width_cell_count: usize,

    selected_cell: Option<GridPosition>, // selected cell (if needed)
    selected_bg_color: Option<macroquad::color::Color>,
}

#[derive(Default, Clone)]
pub struct ColumnSetting {
    pub width: Option<f32>,
}

impl Default for Grid {
    fn default() -> Self {
        Grid::new(0.0, 0.0, 0, 0, 0.0)
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug)]
pub struct Dimensions {
    pub rows: usize,
    pub cols: usize,
}

impl Dimensions {
    pub fn max(self, other: Self) -> Self {
        Self {
            rows: self.rows.max(other.rows),
            cols: self.cols.max(other.cols),
        }
    }
    pub fn from_wh(width: usize, height: usize) -> Self {
        Self {
            rows: height,
            cols: width,
        }
    }
    pub fn width(&self) -> usize {
        self.cols
    }
    pub fn height(&self) -> usize {
        self.rows
    }
}

impl Grid {
    /// position the grid somewhere on the screen
    pub fn set_x_offset(&mut self, x_offset: position::Position) {
        self.x_offset = x_offset;
    }

    /// position the grid somewhere on the screen
    pub fn set_y_offset(&mut self, y_offset: position::Position) {
        self.y_offset = y_offset;
    }

    pub fn set_column_width(&mut self, col: usize, width: impl Into<Option<f32>>) {
        let width = width.into();
        if width == self.column_settings[col].width {
            return;
        }
        let increase = width.unwrap_or(0.0) - self.column_settings[col].width.unwrap_or(0.0);
        self.fixed_width_total += increase;
        self.fixed_width_cell_count +=
            width.is_some() as usize - self.column_settings[col].width.is_some() as usize;
        self.column_settings[col].width = width;
        // Invalidate cache
        self.running_totals_before.borrow_mut().clear();
    }

    pub fn resize(&mut self, width: impl Into<Option<usize>>, height: impl Into<Option<usize>>) {
        let width = width.into().unwrap_or(self.cols);
        let height = height.into().unwrap_or(self.rows);
        if Dimensions::from_wh(width, height) != self.dimensions() {
            self.column_settings.resize_with(width, Default::default);
            self.running_totals_before.borrow_mut().clear();
            self.cells
                .resize_with(height, || vec![Default::default(); width]);
            for row in self.cells.iter_mut() {
                row.resize_with(width, Default::default);
            }
            self.cols = width;
            self.rows = height;
            if let Some(selected) = self.selected_cell {
                if self.get(selected).is_none() {
                    self.selected_cell = None;
                }
            }
        }
    }

    pub fn dimensions(&self) -> Dimensions {
        Dimensions::from_wh(self.cols, self.rows)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn height(&self) -> f32 {
        self.height
    }
    pub fn width(&self) -> f32 {
        self.width
    }

    pub fn set_width(&mut self, width: f32) {
        if self.width == width {
            return;
        }
        self.width = width;
        // Invalidate cache
        self.running_totals_before.borrow_mut().clear();
    }
    pub fn set_height(&mut self, height: f32) {
        self.height = height;
    }

    /// # create a grid
    ///
    /// ## problems
    ///
    /// there are a shit ton of feilds and I wanted the new function
    /// to not have a trillion args.
    /// It is "normal" (more like intended) to create a new Grid and then call a bunch of setters to customize it
    /// to your liking
    pub fn new(width: f32, height: f32, cols: usize, rows: usize, gap: f32) -> Self {
        Grid {
            width,
            height,
            cols,
            rows,
            cell_bg_color: WHITE,
            gap,
            gap_color: BLACK,
            selected_cell: None,
            selected_bg_color: Some(BLUE),
            cells: vec![vec![Default::default(); cols]; rows],
            column_settings: vec![Default::default(); cols],
            running_totals_before: RefCell::new(vec![]),
            x_offset: position::Position::default(),
            y_offset: position::Position::default(),
            fixed_width_total: 0.0,
            fixed_width_cell_count: 0,
            auto_resize_text: true,
        }
    }

    // returns the (width, height) of each cell
    fn calculate_dimensions(&self) -> (f32, f32) {
        // in pixels
        let total_x_gap_space = (self.cols + 1) as f32 * self.gap;
        let total_y_gap_space = (self.rows + 1) as f32 * self.gap;

        let non_fixed_cell_count = self.cols - self.fixed_width_cell_count;
        let cell_width = if non_fixed_cell_count > 0 {
            (self.width - self.fixed_width_total - total_x_gap_space as f32).max(0.0)
                / non_fixed_cell_count as f32
        } else {
            0.0
        };
        let cell_height = (self.height - total_y_gap_space as f32).max(0.0) / self.rows as f32;

        (cell_width, cell_height)
    }

    /// # draw it!
    /// this does not change any state
    /// your gonna want to put this in the main
    /// loop or something like that
    pub fn draw(&self) {
        // draw background (the gap color)
        let layout_calc = self.layout_calculator();
        draw_rectangle(
            layout_calc.offset.x,
            layout_calc.offset.y,
            self.width,
            self.height,
            self.gap_color,
        );

        // draw cells

        for row in 0..self.rows {
            for col in 0..self.cols {
                let pos = GridPosition { row, col };
                self.draw_cell(pos, layout_calc.cell_inner(pos));
            }
        }
        // draw_rectangle_lines(x_offset, y_offset, self.width, self.height, 1.0, BLACK);
    }

    pub fn layout_calculator(&self) -> LayoutCalculator<'_, Ref<'_, [f32]>> {
        let x_offset = self.x_offset.as_pixels(self.width, screen_width());
        let y_offset = self.y_offset.as_pixels(self.height, screen_height());
        let (cell_width, cell_height) = self.calculate_dimensions();
        self.recalculate_running_totals(cell_width);
        LayoutCalculator {
            gap: self.gap,
            cell: Vec2::new(cell_width, cell_height),
            offset: Vec2::new(x_offset, y_offset),
            dim: Vec2::new(self.width, self.height),
            running_totals_before: Ref::map(self.running_totals_before.borrow(), |x| x.as_slice()),
            column_settings: Cow::Borrowed(&self.column_settings),
        }
    }

    fn recalculate_running_totals(&self, cell_width: f32) {
        // Fast exit. Not necessary
        if !self.running_totals_before.borrow().is_empty() {
            return;
        }
        self.running_totals_before
            .borrow_mut()
            .extend(running_totals_from_settings(
                self.column_settings.iter().map(|x| x.width.clone()),
                self.gap,
                cell_width,
                self.width,
            ));
    }

    // Returns a list of all of the segments horizontally.
    pub fn x_boundaries<'a>(&'a self, cell_width: f32) -> impl Iterator<Item = f32> + 'a {
        running_totals_from_settings(
            self.column_settings.iter().map(|x| x.width.clone()),
            self.gap,
            cell_width,
            self.width,
        )
    }

    // only called from the double for loop in the draw function
    // this way it does not look crouded as fuck
    //
    // this function calculates the cells position (takes gap into account)
    // it also handles any special coloring that might need to happen
    // it also prints any text to the screen (if applicable)
    fn draw_cell(&self, cell_pos: GridPosition, cell_rect: Rect) {
        let GridPosition { row, col } = cell_pos;
        let cell = &self.cells[row][col];

        // cell color
        let bg_color = self
            .selected_bg_color
            .filter(|_| self.selected_cell == Some(cell_pos))
            .or_else(|| cell.bg_color)
            .unwrap_or(self.cell_bg_color);

        // draw background rect
        draw_rectangle(cell_rect.x, cell_rect.y, cell_rect.w, cell_rect.h, bg_color);

        // draw the text if this cell has any
        let text = &cell.text;
        if !text.is_empty() {
            // shifted because read the readme
            let y_pos = cell_rect.y + cell_rect.h;

            // Initial guess for the text height
            let mut text_height = cell_rect.h;
            let mut text = text.as_str();
            loop {
                // Check if it fits. If it doesn't then retry using whichever strategy we have.
                let text_dim = macroquad::text::measure_text(text, None, text_height as u16, 1.0); // 1.0 is default
                if self.auto_resize_text && text_dim.width > cell_rect.w {
                    text_height *= cell_rect.w / text_dim.width;
                    text_height = text_height.floor();
                    continue;
                } else if text_dim.width > cell_rect.w {
                    text = truncate_text_single_line(text, &cell_rect, &text_dim);
                    continue;
                }

                let centered_x = (cell_rect.w - text_dim.width) / 2.0 + cell_rect.x;
                let centered_y = y_pos - (cell_rect.h - text_dim.height) / 2.0;

                draw_text(
                    text,
                    centered_x,
                    centered_y,
                    text_height,
                    cell.text_color.unwrap_or(BLACK),
                );
                break;
            }
        }
    }

    // pub fn select_from_mouse(&mut self) -> Option<(usize, usize)> {
    pub fn select_from_mouse(&mut self) -> Option<ClickTarget> {
        let result = self.mouse_hovered_cell()?;
        self.select_cell(result.cell_inner());
        Some(result)
    }

    // pub fn mouse_hovered_cell(&mut self) -> Option<(usize, usize)> {
    pub fn mouse_hovered_cell(&self) -> Option<ClickTarget> {
        self.translate_click(mouse_position().into())
    }

    // pub fn translate_click(&mut self, pos: Vec2) -> Option<(usize, usize)> {
    pub fn translate_click(&self, pos: Vec2) -> Option<ClickTarget> {
        self.layout_calculator().translate_click(pos)
    }

    /// # select a cell
    ///
    /// ## warning
    /// if the selected cell is out of bounds
    /// it might lead to a panic later
    pub fn select_cell(&mut self, cell_index: Option<GridPosition>) {
        self.selected_cell = cell_index;
    }

    /// returns the (row, col) index of the selected cell
    pub fn get_selected_cell_index(&self) -> Option<GridPosition> {
        let selected = self.selected_cell?;
        let _ = self.get(selected)?;
        Some(selected)
    }

    pub fn set_cell_bg(
        &mut self,
        pos: GridPosition,
        color: impl Into<Option<macroquad::color::Color>>,
    ) {
        self.cell_mut(pos).set_bg(color)
    }

    /// # sets default bg color for all cells
    ///
    /// different from color_cell becuase this one applies to all
    /// uncolored and unselected cells
    /// this function panics
    pub fn set_cell_bg_color(&mut self, color: macroquad::color::Color) {
        self.cell_bg_color = color;
    }

    /// color the gap between cells
    pub fn set_gap_color(&mut self, color: macroquad::color::Color) {
        self.gap_color = color;
    }

    /// when selected, a cell will have this color
    pub fn set_selected_cell_color(&mut self, color: macroquad::color::Color) {
        self.selected_bg_color = Some(color);
    }

    /// # write text to a cell
    ///
    /// ## panics
    /// if row and col are out of bounds
    ///
    /// ## generic option
    /// so the text arg is the text to be written
    /// - if the Option is None, there will be no text
    /// - if the Option is Some(text), I call text.to_string()
    /// and then write the resulting String to the screen
    pub fn set_cell_text<T>(&mut self, pos: GridPosition, text: impl Into<Option<T>>)
    where
        T: std::fmt::Display,
    {
        self.cell_mut(pos).set_text(text)
    }

    pub fn cell_mut(&mut self, GridPosition { row, col }: GridPosition) -> &mut cell::Cell {
        &mut self.cells[row][col]
    }

    pub fn cell(&self, GridPosition { row, col }: GridPosition) -> &cell::Cell {
        &self.cells[row][col]
    }

    pub fn get(&self, GridPosition { row, col }: GridPosition) -> Option<&cell::Cell> {
        self.cells.get(row)?.get(col)
    }

    pub fn get_mut(&mut self, GridPosition { row, col }: GridPosition) -> Option<&mut cell::Cell> {
        self.cells.get_mut(row)?.get_mut(col)
    }

    /// same as set_cell_text
    /// but instead of providing a row and col
    /// it just writes the text onto the selected cell
    ///
    /// ## no selected cell
    ///
    /// if there is no selected cell, this
    /// method does nothing
    ///
    /// ## panics
    ///
    /// if the selected cell happens to be out of bounds,
    /// this function panics
    pub fn set_selected_cell_text<T>(&mut self, text: impl Into<Option<T>>)
    where
        T: std::fmt::Display,
    {
        // only do something if there is a selected cell
        if let Some(pos) = self.get_selected_cell_index() {
            self.cell_mut(pos).set_text(text)
        }
    }
}

fn truncate_text_single_line<'a>(
    text: &'a str,
    cell_rect: &Rect,
    text_dim: &TextDimensions,
) -> &'a str {
    let char_count = text.chars().count();
    let mut it = text.chars();
    for _ in 0..(char_count as f32 * cell_rect.w / text_dim.width) as usize {
        it.next();
    }
    &text[..text.len() - it.as_str().len()]
}

impl Index<GridPosition> for Grid {
    type Output = cell::Cell;

    fn index(&self, index: GridPosition) -> &Self::Output {
        self.cell(index)
    }
}

impl IndexMut<GridPosition> for Grid {
    fn index_mut(&mut self, index: GridPosition) -> &mut Self::Output {
        self.cell_mut(index)
    }
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GridVar<T> {
    pub row: T,
    pub col: T,
}

pub type GridPosition = GridVar<usize>;

impl GridPosition {
    pub fn advance_row(mut self, count: usize) -> Self {
        self.row += count;
        self
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ClickTarget {
    pub cell: GridPosition,
    pub is_border: GridVar<bool>,
    pub is_outer_min: GridVar<bool>,
    pub is_outer_max: GridVar<bool>,
}

impl GridVar<bool> {
    pub fn any(&self) -> bool {
        self.row || self.col
    }
}

impl ClickTarget {
    pub fn cell_inner(self) -> Option<GridPosition> {
        if self.is_border.any() || self.is_outer_min.any() || self.is_outer_max.any() {
            return None;
        }
        Some(self.cell)
    }
}

pub struct LayoutCalculator<'a, T = &'a [f32]> {
    pub cell: Vec2,
    pub offset: Vec2,
    pub gap: f32,
    pub dim: Vec2,
    running_totals_before: T,
    column_settings: Cow<'a, [ColumnSetting]>,
}

fn gapped_cell_offset(gap: f32, length: f32, offset: usize) -> f32 {
    gap + offset as f32 * (length + gap)
}

pub fn running_totals_from_settings(
    widths: impl Iterator<Item = Option<f32>>,
    // settings: &[ColumnSetting],
    gap: f32,
    cell_width: f32,
    total_width: f32,
) -> impl Iterator<Item = f32> {
    widths
        .scan(0.0, move |total, width| {
            *total += gap;
            let gap_end = *total;
            let width = width.unwrap_or(cell_width);
            *total += width;
            let cell_end = *total;
            Some([gap_end, cell_end])
        })
        .flatten()
        .chain(std::iter::once(total_width))
}

pub struct FixedLayoutSettings {
    pub cell: Vec2,
    pub offset: Vec2,
    pub gap: f32,
    pub dim: Vec2,
}

impl<'a> LayoutCalculator<'a, Vec<f32>> {
    pub fn from_settings(
        FixedLayoutSettings {
            cell,
            offset,
            gap,
            dim,
        }: FixedLayoutSettings,
        settings: impl Into<Cow<'a, [ColumnSetting]>>,
    ) -> Self {
        let column_settings = settings.into();
        Self {
            cell,
            offset,
            gap,
            dim,
            running_totals_before: running_totals_from_settings(
                column_settings.iter().map(|x| x.width.clone()),
                gap,
                cell.x,
                dim.x,
            )
            .collect(),
            column_settings,
        }
    }
}

impl<T> LayoutCalculator<'_, T>
where
    T: std::ops::Deref<Target = [f32]>,
{
    pub fn rect(&self) -> Rect {
        Rect::new(self.offset.x, self.offset.y, self.dim.x, self.dim.y)
    }

    pub fn cell_inner(&self, GridPosition { row, col }: GridPosition) -> Rect {
        // let x_pos = self.running_totals_before[col * 2];
        let x_pos = self
            .running_totals_before
            .get(col * 2)
            .copied()
            .unwrap_or_else(|| gapped_cell_offset(self.gap, self.cell.x, col));
        let y_pos = gapped_cell_offset(self.gap, self.cell.y, row);
        // let cell_width = self.column_settings[col]
        //     .width
        //     .unwrap_or(self.cell.x)
        //     .min(self.dim.x - self.gap - x_pos)
        //     .max(0.0);
        let cell_width = self
            .column_settings
            .get(col)
            .and_then(|setting| setting.width)
            .unwrap_or(self.cell.x)
            .min(self.dim.x - self.gap - x_pos)
            .max(0.0);
        let cell_height = self.cell.y.min(self.dim.y - self.gap - y_pos).max(0.0);
        Rect {
            x: x_pos,
            y: y_pos,
            w: cell_width,
            h: cell_height,
        }
        .offset(self.offset)
    }

    pub fn translate_click(&self, pos: Vec2) -> Option<ClickTarget> {
        if !self.rect().contains(pos) {
            return None;
        }
        let pos = pos - self.offset;
        let is_outer_min = GridVar {
            col: pos.x < self.gap,
            row: pos.y < self.gap,
        };
        let is_outer_max = GridVar {
            col: pos.x > (self.dim.x - self.gap),
            row: pos.y > (self.dim.y - self.gap),
        };
        if is_outer_min.any() || is_outer_max.any() {
            return Some(ClickTarget {
                cell: Default::default(),
                is_border: Default::default(),
                is_outer_min,
                is_outer_max,
            });
        }
        let (Ok(col) | Err(col)) = self
            .running_totals_before
            .binary_search_by(|end| end.partial_cmp(&pos.x).unwrap());
        let is_col_border = (col & 1) == 0;
        let col = col / 2;

        let cell_height_with_gap = self.cell.y + self.gap;
        let row = (pos.y - self.gap) / cell_height_with_gap;
        let is_row_border = pos.y > ((row.floor() + 1.0) * cell_height_with_gap);
        let row = row as usize;
        Some(ClickTarget {
            cell: GridPosition { row, col },
            is_border: GridVar {
                row: is_row_border,
                col: is_col_border,
            },
            is_outer_min: Default::default(),
            is_outer_max: Default::default(),
        })
    }
}
