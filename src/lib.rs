#[cfg(test)]

extern crate float_cmp;
extern crate chrono;
// extern crate date_time;

pub mod cashflow;
pub mod interest_rate;
pub mod daycounter;

mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
