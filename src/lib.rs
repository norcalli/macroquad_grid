//! # a grid to use with the macroquad lib
//!
//! so if you are:
//! - making a sudoku game
//! - chess game
//! - battleship
//! - etc.
//!
//! you will probably want a grid to work with
//!
//! This is a grid to use with macroquad!
//!
//! There is a struct called Grid and an enum that you can
//! use if you want the grid to be somewhere other than the top left corner
//! (the grids top left corner is the top left corner of the screen unless you
//! use the enum + setter on grid to move it)
//!
//! ## this crate NEEDS to be used with macroquad
//! its an addon! AN ADDON
//! you use the grid in your macroquad program
//!
//! ## stuff you can do with the grid
//! in src/grid/main.rs I call every method on the grid
//! struct so that should be helpful
//!
//! Most of them are setters so it should be pretty
//! straight forward
//!
//! ## summary
//!
//! you can make a grid and then color the cells, write text to the cells, yeah
//!
//! ## cringe
//!
//! so a lot of the set_color methods may seems pretty similar
//! I promise they do not all do the same thing.
//!
//! ### elaborate!
//!
//! cells have a default bg color and a default selected color
//! each of these defaults can be overwritten with one of the setters
//!
//! you can also explicitly color a cell with a third setter
//!
//! the gap between cells can also be colored with a setter
//!
//! ## panic
//!
//! so when you create a grid it will have a width and a height
//! (set by you if you use the new method on Grid)
//! I create a 2D vector with height inner vectors
//! and each inner vector has width cells
//! if you try to select, write to, color, etc. a cell
//! at a row or col that is bigger than the width or
//! height respectively, then this crate will panic
//!
//! ### tldr
//! don't color/write to/set to a cell that does not exist/is out of bounds
//!
pub mod grid;

pub use grid::Grid;
pub use grid::Position;

#[cfg(test)]
mod tests {
    //use super::*;
    #[test]
    fn it_works() {
        todo!()
    }
}
