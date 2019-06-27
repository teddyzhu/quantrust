use chrono::prelude::*;
type Real = f32;
type LocalDate = Date<Local>;

pub trait DayCounter{
    fn name(&self)->&str;
    fn day_count(&self, d1:&LocalDate, d2:&LocalDate)->i32{
       days_between(d1, d2)
    }
    fn year_fraction(&self, d1:&LocalDate, d2:&LocalDate,
                    refStart:&LocalDate, refEnd:&LocalDate)->Real;
}

#[derive(Clone)]
pub enum DayCounterEnum{
    Actual360,
    Actual365Fixed
}

impl DayCounterEnum{
    fn days_between(d1:&LocalDate, d2:&LocalDate)->i32{
        d2.num_days_from_ce()-d1.num_days_from_ce()
    }

    fn name(&self)->&str{
        match self{
            DayCounterEnum::Actual360=>"Actual/360",
            DayCounterEnum::Actual365Fixed=>"Actual/365 (Fixed)",
        }
    }
    fn year_fraction(&self, d1:&LocalDate, d2:&LocalDate,
                    refStart:&LocalDate, refEnd:&LocalDate)->f32{
        match self{
             DayCounterEnum::Actual360=>Self::days_between(d1, d2) as f32 /360.0,
             DayCounterEnum::Actual365Fixed=>Self::days_between(d1, d2) as f32 /365.0,
        }
    }
}

fn days_between(d1:&LocalDate, d2:&LocalDate)->i32{
    d2.num_days_from_ce()-d1.num_days_from_ce()
}

#[derive(Clone)]
pub struct Actual365Fixed;

impl DayCounter for Actual365Fixed{
    fn name(&self)->&str{
        "Actual/365 (Fixed)"
    }
    fn year_fraction(&self, d1:&LocalDate, d2:&LocalDate,
                    refStart:&LocalDate, refEnd:&LocalDate)->f32{
        days_between(d1, d2) as f32 /365.0
    }
}
impl std::fmt::Debug for Actual365Fixed{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result{
        write!(f, "({:?}", self)
 }
}

struct MyDc{
    dc_ : Option<Box<DayCounter>>,
}

#[test]
fn test_dc(){
    let dc = DayCounterEnum::Actual360;
    eprintln!("dc name: {}, year_frac: {}", dc.name(), 
        dc.year_fraction(&Local.ymd(2018,1,1), &Local.ymd(2019,1,1), 
                        &Local.ymd(2018,1,1), &Local.ymd(2018,1,1)));
    eprintln!("dc fix name:{}, year_frac:{}", DayCounterEnum::Actual365Fixed.name(),
        DayCounterEnum::Actual365Fixed.year_fraction(&Local.ymd(2018,1,1), &Local.ymd(2019,1,1), 
                        &Local.ymd(2018,1,1), &Local.ymd(2018,1,1)));

    let dc = Actual365Fixed;
    let mydc = MyDc{dc_:Option::Some(Box::new(dc))};
}