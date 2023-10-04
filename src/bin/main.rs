use macroquad::prelude::*;

// the grid!
use macroquad_grid::Grid;

// optional enum used to postion the grid
use macroquad_grid::Position;

// this file shows off
// - creating a grid
// - selecting a cell
// - changing selected cells color
// - changing default cell bg color
// - changing gap color
// - changing grids postion with Position enum
// - setting color of a specific cell
// - writing text to a specific cell
// - writing text to the selected cell
// - getting the selected cell's index
// - drawing the grid

// just to note, I call all of these methods
// outside of the loop, but you can call them whereever
// (not in a different thread)
// 
// so if you want to update the grid based on user input
// or something, go crazy

#[macroquad::main("BasicShapes")]
async fn main() {
    // create a grid (also can use default)
    let mut g = Grid::new(300.0, 300.0, 10, 10, 1.0);

    // selects the cell at row 4 col 5 (None deselects cell)
    g.select_cell(Some((4, 5)));

    // set the color of the selected cell (if any)
    g.set_selected_cell_color(RED);

    // set the default cell color
    g.set_cell_bg_color(BROWN);

    // sets the gap color to yellow
    g.set_gap_color(YELLOW);

    // you can center the grid or postion it at the top, left, bottom, or right of the screen
    // OR a custom offset with pixels
    g.set_x_offset(Position::Center);
    g.set_y_offset(Position::Center);

    // color a specific cell
    g.color_cell(7, 7, ORANGE);

    // write text to some cell
    g.set_cell_text(0, 0, Some("hi"));

    // write text to selected cell
    g.set_selected_cell_text(Some("sel"));

    // you can get the index of the selected cell
    let i = g.get_selected_cell_index().expect("selected it ~10 lines ago, I know its the some variant");
    println!("{:#?} is the selected cell", i);

    loop {
        // draws the grid
        g.draw();

        next_frame().await
    }
}
