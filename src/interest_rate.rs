#[derive(Debug, Clone, Copy)]
pub enum Compounding{
    Simple,
    Compounded,
    Continuous,
    SimpleThenCompounded,
    CompoundedThenSimple
}

#[derive(Debug, Clone, Copy)]
pub enum Frequency { 
    NoFrequency = -1,     //< null frequency
    Once = 0,             //< only once, e.g., a zero-coupon
    Annual = 1,           //< once a year
    Semiannual = 2,       //< twice a year
    EveryFourthMonth = 3, //< every fourth month
    Quarterly = 4,        //< every third month
    Bimonthly = 6,        //< every second month
    Monthly = 12,         //< once a month
    EveryFourthWeek = 13, //< every fourth week
    Biweekly = 26,        //< every second week
    Weekly = 52,          //< once a week
    Daily = 365,          //< once a day
    OtherFrequency = 999  //< some other unknown frequency
}

use chrono::prelude::*;
type Real = f64;
type LocalDate = Date<Local>;

pub struct InterestRate<'a>{
    pub r_: Real,
    pub dc_: &'a Box<DayCounter>,
    pub comp_: Compounding,
    // freqMakesSense_: bool,
    pub freq_: Frequency
}

impl <'a>InterestRate<'a>{

    pub fn compound_factor(&self, time:Real)->Real{
        let freq_ = self.freq_ as i32 as Real;
        match &self.comp_{
            Compounding::Simple =>{
                1.0 + self.r_ * time
            },
            Compounding::Compounded =>{
                (1.0 + self.r_ / freq_).powf(freq_ * time)
            },
            Compounding::Continuous =>{
               (self.r_ * time).exp()
            },
            Compounding::SimpleThenCompounded =>{
                 if time <= 1.0 / freq_ {
                    1.0+self.r_ * time 
                }else{
                    (1.0+self.r_ / freq_).powf(freq_ * time)
                }
            }
            Compounding::CompoundedThenSimple => unimplemented!(),
        }
    }

    pub fn compound_factor1(&self, d1:&LocalDate, d2:&LocalDate, 
                        refStart:&LocalDate, refEnd:&LocalDate)->Real{
        // unimplemented!()
        assert!(d2 >= d1);
        let t = self.dc_.year_fraction(d1, d2, refStart, refEnd);
        self.compound_factor(t)
        
    }

    pub fn discount_factor(&self, time:Real)->Real{
        1.0/self.compound_factor(time)
    }
}

fn days_between(d1:&LocalDate, d2:&LocalDate)->i32{
    d2.num_days_from_ce()-d1.num_days_from_ce()
}

pub trait DayCounter{
    fn name(&self)->&str;
    fn day_count(&self, d1:&LocalDate, d2:&LocalDate)->i32{
       days_between(d1, d2)
    }
    fn year_fraction(&self, d1:&LocalDate, d2:&LocalDate,
                    refStart:&LocalDate, refEnd:&LocalDate)->Real;
}

#[derive(Clone)]
pub struct Actual360;

impl DayCounter for Actual360{
    fn name(&self)->&str{
        "Actual/360"
    }
    fn year_fraction(&self, d1:&LocalDate, d2:&LocalDate,
                    refStart:&LocalDate, refEnd:&LocalDate)->Real{
        days_between(d1, d2) as Real /360.0
    }
}

#[derive(Clone)]
pub struct Actual365Fixed;

impl DayCounter for Actual365Fixed{
    fn name(&self)->&str{
        "Actual/365 (Fixed)"
    }
    fn year_fraction(&self, d1:&LocalDate, d2:&LocalDate,
                    refStart:&LocalDate, refEnd:&LocalDate)->Real{
        days_between(d1, d2) as Real /365.0
    }
}

pub struct TermStructure{
    referenceDate: LocalDate,
    settlementDays: LocalDate,
    dayCounter: DayCounter,
}


#[test]
fn test_int(){
    let dc = Box::new(Actual360) as Box<DayCounter>;
    let interest = InterestRate{
        r_: 0.10,
        dc_: &dc,
        comp_: Compounding::Compounded,
        freq_ : Frequency::Monthly
    };
    eprintln!("compound_factor: {}", interest.compound_factor(1.0));
    eprintln!("discount_factor: {}", interest.discount_factor(1.0));
}