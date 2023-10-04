#[derive(Clone, Copy, Default)]
/// # position the grid
///
/// sort of like bad css styles
///
/// ## note
///
/// this enum works for both x and y axis
/// its only use is to be passed into the set_x_offset and
/// set_y_offset Grid methods.
/// you can use it to center the grid and other stuff
///
/// ## Start
///
/// the start variant represents the top or left end of the screen
/// that is, an offset of 0.0!
///
/// ## End
///
/// the end offset represents the bottom or right end of the screen
///
/// ## Center
///
/// What to you think? it centers the grid
///
/// ## Pixels
///
/// If you would like to offset the grid by a custom amount, use this variant.
/// - positive values represent down and right
/// - negative values represent left or up
///
/// ## examples
///
/// I am gonna list some (x, y) tuples where x and y are variants
/// of this enum.
/// here, (x, y) is shorthand for calling:
/// grid.set_x_offset(x) and then
/// grid.set_y_offset(y)
///
/// - so (Start, Start) is the top left corner
/// - (Start, End) is bottom left corner
/// - (End, Center) is the middle-right edge of the screen
/// - (Center, Start) is the top-middle of the screen
/// - (Pixels(100), End) is at the bottom of the screen, 100 Pixels to the right
/// - (End, Pixels(100)) is at the right of the screen, 100 pixels from the top
///
pub enum Position {
    #[default]
    Start, // left or top
    End,         // right or bottom
    Center,      // middle (either way)
    Pixels(f32), // + means right or down and - means left or up
    Percent(f32),
}

impl Position {
    pub fn as_pixels(self, width_or_height_of_thing: f32, width_or_height_of_screen: f32) -> f32 {
        as_pixels(self, width_or_height_of_thing, width_or_height_of_screen)
    }
}

pub fn as_pixels(
    position: Position,
    width_or_height_of_thing: f32,
    width_or_height_of_screen: f32,
) -> f32 {
    match position {
        Position::Start => 0.0,
        Position::End => width_or_height_of_screen - width_or_height_of_thing,
        Position::Center => (width_or_height_of_screen - width_or_height_of_thing) / 2.0,
        Position::Pixels(offset) => offset,
        Position::Percent(percent) => {
            (width_or_height_of_screen - width_or_height_of_thing) * percent / 100.0
        }
    }
}

impl From<f32> for Position {
    fn from(value: f32) -> Self {
        Position::Pixels(value)
    }
}
