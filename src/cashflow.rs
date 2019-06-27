extern crate chrono;

use chrono::prelude::*;

// use float_cmp::F32Margin;
use float_cmp::F64Margin;

use float_cmp::ApproxEq;
use float_cmp::approx_eq;


use crate::interest_rate::{
    InterestRate,
    Frequency,
    Compounding,
    DayCounter,
    Actual360,Actual365Fixed,
    TermStructure
};

type Real = f64;
type LocalDate = Date<Local>;
trait Observable{
}

trait AcyclicVisitor{

}

// trait Event: Observable{
pub trait Event{
    fn date(&self)->&LocalDate;

    fn has_occurred(&self, refDate:&LocalDate, includeRefDate:bool)->bool{
        let refDate_ = if *refDate != Local::today(){*refDate}else{
                unsafe{
                    SETTINGS.evaluationDate_.expect("no setting evaluation date")
                }
            };
        let includeRefDateEvent = if includeRefDate {includeRefDate}else{
                unsafe{
                    SETTINGS.includeReferenceDateEvents_
                }
            };
        if includeRefDateEvent {
            self.date() < refDate
        } else{
            self.date() <= refDate
        }
    }

    // fn accept(v:&AcyclicVisitor){
    // }
}

pub trait CashFlow:Event+std::fmt::Debug{
    fn amount(&self)->Real;
    fn exCouponDate(&self)->LocalDate{
        Local::now().date()
    }
    fn trading_ex_coupon(&self, refDate: &LocalDate)->bool{
        false
    }
    fn get_stepwise_discount_time(&self, dc:&Box<dyn DayCounter>,
                            npvDate:&LocalDate,
                            lastDate:&LocalDate)->Real{
        let cashflow_date = &self.date();
        let refStartDate:LocalDate;
        let refEndDate:&LocalDate;
        if lastDate == npvDate {
            refStartDate = cashflow_date.with_year(cashflow_date.year()+1).expect("invalid year set up.");
        } else  {
            refStartDate = *lastDate;
        }
        refEndDate = cashflow_date;
        eprintln!("refstart date:{}, lastDate:{}, cashflow_date:{}", refStartDate, lastDate, cashflow_date);
        return dc.year_fraction(lastDate, cashflow_date, &refStartDate, refEndDate);
    }
}

// #[derive(Debug, Clone)]
struct SimpleCashFlow{
    amount_:Real,
    date_:LocalDate
}

impl Event for SimpleCashFlow {
    fn date(&self)->&LocalDate{
        &self.date_
    }
}

impl CashFlow for SimpleCashFlow{
    fn amount(&self)->Real{
        self.amount_
    }
}

trait Coupon:CashFlow{
    fn nominal(&self)->Real;
    fn rate(&self)->&InterestRate;
    fn dayCounter(&self)->&Box<dyn DayCounter>;
    fn accruedAmount(&self, d:&LocalDate)->Real;
}

struct FixedRateCoupon<'a>{
    paymentDate_:LocalDate,
    nominal_:Real,
    accrualStartDate_:LocalDate,accrualEndDate_:LocalDate,
    refPeriodStart_:LocalDate, refPeriodEnd_:LocalDate,
    exCouponDate_:LocalDate,
    accrualPeriod_:Real,
    rate_:InterestRate<'a>,
}

impl Event for FixedRateCoupon<'_> {
    fn date(&self)->&LocalDate{
        &self.paymentDate_
    }
}

impl Coupon for FixedRateCoupon<'_>{
    fn nominal(&self)->Real{
        self.nominal_
    }
    fn rate(&self)->&InterestRate{
        &self.rate_
    }
    fn dayCounter(&self)->&Box<dyn DayCounter>{
        &self.rate_.dc_
    }
    fn accruedAmount(&self, d:&LocalDate)->Real{
         if d <= &self.accrualStartDate_ || d > &self.paymentDate_ {
            return 0.0
        } else if self.trading_ex_coupon(d) {
            return -self.nominal()*(self.rate_.compound_factor1(d,
                                                    &self.accrualEndDate_,
                                                    &self.refPeriodStart_,
                                                    &self.refPeriodEnd_) - 1.0)
        } else {
            return self.nominal()*(self.rate_.compound_factor1(&self.accrualStartDate_,
                                                   &d.min(&self.accrualEndDate_),
                                                   &self.refPeriodStart_,
                                                   &self.refPeriodEnd_) - 1.0)
        }
    }
}

impl CashFlow for FixedRateCoupon<'_>{
    fn amount(&self)->Real{
        // let coupon = self as &Coupon;
        self.nominal()*(self.rate_.compound_factor1(&self.accrualStartDate_,
                                               &self.accrualEndDate_,
                                               &self.refPeriodStart_,
                                               &self.refPeriodEnd_) - 1.0)
    }

    fn get_stepwise_discount_time(&self, dc:&Box<dyn DayCounter>,
                            npvDate:&LocalDate,
                            lastDate:&LocalDate)->Real{
        if lastDate != &self.accrualStartDate_ {
            let couponPeriod = dc.year_fraction(&self.accrualStartDate_,
                                        &self.date(), &self.refPeriodStart_, &self.refPeriodEnd_);
            let accruedPeriod = dc.year_fraction(&self.accrualStartDate_,
                                        lastDate, &self.refPeriodStart_, &self.refPeriodEnd_);
            return couponPeriod - accruedPeriod
        } else {
            return dc.year_fraction(lastDate, &self.date(),
                            &self.refPeriodStart_, &self.refPeriodEnd_);
        }
    }
}

use std::fmt;

impl fmt::Debug for FixedRateCoupon<'_>{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Coupon {{ date: {}, amount: {} }}", self.date(), self.amount())
    }
}


impl fmt::Debug for SimpleCashFlow{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Simple CashFlow {{ date: {}, amount: {} }}", self.date(), self.amount())
    }
}

type Leg = Vec<Box<dyn CashFlow>>;



fn npv(leg:&Leg, 
    y:&InterestRate, 
    includeSettlementDateFlows:bool, 
    mut settlementDate:LocalDate,
    mut npvDate:LocalDate)->Real{
        if leg.is_empty(){
            return 0.0;
        }

        if settlementDate == Local::today(){
            unsafe{
            settlementDate = SETTINGS.evaluationDate_.expect("no set evaluation date.");
            }
        }

        if npvDate == Local::today(){
            npvDate = settlementDate;
        }

        let mut npv = 0.0;
        let mut discount = 1.0;
        let mut lastDate = &npvDate;
        let dc = y.dc_;
        // for (Size i=0; i<leg.size(); ++i) {
        let mut i = 0;
        while i < leg.len(){
            if leg[i].has_occurred(&settlementDate, includeSettlementDateFlows){
                i += 1;
                continue;
            }

            let mut amount = leg[i].amount();
            if leg[i].trading_ex_coupon(&settlementDate) {
                amount = 0.0;
            }

            let b = y.discount_factor(leg[i].get_stepwise_discount_time(dc, &npvDate, &lastDate));
            discount *= b;
            lastDate = leg[i].date();

            npv += amount * discount;
            eprintln!("calc npv : {}", npv);
            i += 1;
        }

        return npv;
}

struct Settings{
    evaluationDate_:Option<LocalDate>,
    includeReferenceDateEvents_:bool,
    includeTodaysCashFlows_:bool,
    enforcesTodaysHistoricFixings_:bool
}

static mut SETTINGS:Settings = Settings{
    evaluationDate_:None,
    includeReferenceDateEvents_:false,
    includeTodaysCashFlows_:false,
    enforcesTodaysHistoricFixings_:false
};

pub fn simple_duration(leg:&Leg,
    y:&InterestRate,
    includeSettlementDateFlows:bool,
    settlementDate:&LocalDate,
    npvDate:&LocalDate
)->Real{
    if leg.is_empty(){ return 0.0; }

    // if (settlementDate == Date())
    //     settlementDate = Settings::instance().evaluationDate();

    // if (npvDate == Date())
    //     npvDate = settlementDate;

    let mut P = 0.0;
    let mut dPdy = 0.0;
    let mut t = 0.0;
    let mut lastDate = npvDate;
    let dc = y.dc_;
    // for (Size i=0; i<leg.size(); ++i) {
    let mut i = 0;
    while i < leg.len() {
        if leg[i].has_occurred(settlementDate, includeSettlementDateFlows){
            i += 1;
            continue;
        }

        let mut c = leg[i].amount();
        if leg[i].trading_ex_coupon(&settlementDate) {
            c = 0.0;
        }

        t += leg[i].get_stepwise_discount_time(dc, &npvDate, &lastDate);
        eprintln!("get t : {}", t);
        let B = y.discount_factor(t);
        eprintln!("get B : {}", B);
        P += c * B;
        eprintln!("get P : {}", P);
        dPdy += t * c * B;
        eprintln!("get dPdy : {}", dPdy);
        
        lastDate = &leg[i].date();
        i += 1;
    }
    if P == 0.0 {// no cashflows
        return 0.0;
    }
    return dPdy/P;
}

fn macaulay_duration(
    leg:&Leg,
    y:&InterestRate,
    includeSettlementDateFlows:bool,
    mut settlementDate:LocalDate,
    mut npvDate:LocalDate)->Real{

//    assert_eq!(y.comp_, &Compounding::Compounded);
    (1.0 + y.r_ / y.freq_ as i32 as Real) 
        * modified_duration(leg, y, includeSettlementDateFlows, settlementDate, npvDate)
}

fn modified_duration(
    leg:&Leg, 
    y:&InterestRate, 
    includeSettlementDateFlows:bool, 
    mut settlementDate:LocalDate,
    mut npvDate:LocalDate
)->Real{
    if leg.is_empty(){return 0.0}
    // if settlementDate == Local::today(){
    //     unsafe{
    //     settlementDate = SETTINGS.evaluationDate_.expect("No set evaluation date.");
    //     }
    // }
    // if npvDate == Local::today(){
    //     npvDate = settlementDate
    // }

    let mut P:Real = 0.0;
    let mut t:Real = 0.0;
    let mut dPdy = 0.0;
    let r = y.r_;
    let N = y.freq_ as i32 as Real;
    let mut lastDate = &npvDate;
    let dc = y.dc_;
    let mut i = 0;
    while i<leg.len(){
        if leg[i].has_occurred(&settlementDate, includeSettlementDateFlows){
            i += 1;
            continue
        };

        let mut c = leg[i].amount();
        if leg[i].trading_ex_coupon(&settlementDate) {
            c = 0.0;
        }

        t += leg[i].get_stepwise_discount_time(dc, &npvDate, &lastDate);
        let B = y.discount_factor(t);
        eprintln!("i:{}, get B:{}", i, B);
        P += c * B;
        match y.comp_ {
            Compounding::Simple => dPdy -= c * B*B * t,
            Compounding::Compounded => dPdy -= c * t * B/(1.0 + r/N),
            Compounding::Continuous => dPdy -= c * B * t,
            Compounding::SimpleThenCompounded =>{
                                                    if t<=1.0/N {
                                                        dPdy -= c * B*B * t}
                                                    else{
                                                        dPdy -= c * t * B/(1.0 + r/N)
                                                    }
                                                },
            Compounding::CompoundedThenSimple=>{
                                                    if t > 1.0/N{
                                                        dPdy -= c * B*B * t
                                                    }else{
                                                        dPdy -= c * t * B/(1.0 + r/N)
                                                    }
                                                }
        }
        lastDate = leg[i].date();
        i += 1;
    }

    if P == 0.0 // no cashflows
    {
        return 0.0;
    }

    eprintln!("dPdy:{}, P:{}", -dPdy, P);

    return -dPdy/P; // reverse derivative sign
}




const growthFactor:f64 = 1.6;
const QL_EPSILON:f64 = std::f64::EPSILON;
const MAX_FUNCTION_EVALUATIONS:usize = 100;




fn close(f1:f64, f2:f64)->bool{
    approx_eq!(f64, f1, f2, F64Margin::default())
}

trait Derivative{
    fn call(&self, y:Real)->Real;
    fn derivative(&self, y:Real)->Real;
}


struct IrrFinder{
    leg_:Leg,
    npv_:Real,
    dayCounter_:Box<DayCounter>,
    compounding_:Compounding,
    frequency_:Frequency,
    includeSettlementDateFlows_:bool,
    settlementDate_:LocalDate, 
    npvDate_:LocalDate
}

impl IrrFinder{
    fn new(leg:Leg, npv:Real, 
        dayCounter:Box<DayCounter>, 
        compounding:Compounding,
        frequency:Frequency,
        includeSettlementDateFlows:bool,
        settlementDate:LocalDate, 
        npvDate:LocalDate)->Box<IrrFinder>{
        Box::new(IrrFinder{
            leg_:leg,
            npv_:npv,
            dayCounter_:dayCounter,
            compounding_:compounding,
            frequency_:frequency,
            includeSettlementDateFlows_:includeSettlementDateFlows,
            settlementDate_:settlementDate, 
            npvDate_:npvDate
        })
    }
}

impl Derivative for IrrFinder{
    fn call(&self, y:Real)->Real{
        let _yield = InterestRate{
            r_: y,
            dc_: &self.dayCounter_,
            comp_ : self.compounding_,
            freq_: self.frequency_
        };
        let _npv = npv(&self.leg_, &_yield, 
            self.includeSettlementDateFlows_, 
            self.settlementDate_, self.npvDate_);
        self.npv_ - _npv

    }
    fn derivative(&self, y:Real)->Real{
        let _yield = InterestRate{
            r_: y,
            dc_: &self.dayCounter_,
            comp_ : self.compounding_,
            freq_: self.frequency_
        };
        modified_duration(&self.leg_, &_yield,
                        self.includeSettlementDateFlows_,
                        self.settlementDate_, self.npvDate_)
    }
}

fn generate_cashflows1()->Leg{
    let leg = vec!(Box::new(SimpleCashFlow{
    amount_:-87680.67,
    date_:Local.ymd(2018, 3, 31),
    }) as Box<CashFlow>,
    Box::new(SimpleCashFlow{
        amount_:41349.55,
        date_:Local.ymd(2018, 4, 26),
    }) as Box<CashFlow>,
    Box::new(SimpleCashFlow{
        amount_:20290.71,
        date_:Local.ymd(2018, 7, 26),
    }) as Box<CashFlow>,
    Box::new(SimpleCashFlow{
        amount_:14438.57,
        date_:Local.ymd(2018, 10, 26),
    }) as Box<CashFlow>,
    Box::new(SimpleCashFlow{
        amount_:14609.82,
        date_:Local.ymd(2019, 1, 26),
    }) as Box<CashFlow>,
    );
    eprintln!("get cashflow {:#?}", leg);
    leg
}

fn generate_cashflows()->Leg{
    let leg = vec!(Box::new(SimpleCashFlow{
    amount_:-100.0,
    date_:Local.ymd(2018, 3, 31),
    }) as Box<CashFlow>,
    Box::new(SimpleCashFlow{
        amount_:110.0,
        date_:Local.ymd(2019, 3, 31),
    }) as Box<CashFlow>,
    );
    eprintln!("get cashflow {:#?}", leg);
    leg
}

#[test]
fn test_derivative(){
    unsafe{
        SETTINGS.evaluationDate_ = Some(Local::today());
    }

    let leg = generate_cashflows1();

    let dc = Box::new(Actual365Fixed) as Box<DayCounter>;
    let interest = InterestRate{
        // r_: 0.1,
        r_: 0.10543711301399399,
        dc_: &dc,
        comp_: Compounding::Compounded,
        freq_ : Frequency::Annual
        // freq_ : Frequency::Quarterly
    };
    let date = Local.ymd(2018, 3, 31);
    let npv_ = npv(&leg, &interest, true, date, date);
    eprintln!("npv is {}", npv_);
    let duration = modified_duration(&leg, &interest, false, date, date);
    let mac_duration = macaulay_duration(&leg, &interest, false, date, date);
    let sim_duration = simple_duration(&leg, &interest, false, &date, &date);
    eprintln!("modified duration is {}, mac duration:{}, simple duration:{}", 
        duration, mac_duration, sim_duration);

    let irrFinder = IrrFinder{
        leg_:leg,
        npv_:0.0,
        dayCounter_:Box::new(Actual360),
        compounding_:Compounding::Continuous,
        frequency_:Frequency::Monthly,
        includeSettlementDateFlows_:true,
        settlementDate_:Local::today(), 
        npvDate_:Local::today(),
    };
    eprintln!("functor: {}", irrFinder.call(0.5));
    eprintln!("derivative: {}", irrFinder.derivative(0.5));
}

#[test]
fn test_irr(){
    unsafe{
        SETTINGS.evaluationDate_ = Some(Local::today());
    }
    let dc = Box::new(Actual365Fixed) as Box<DayCounter>;
    let leg = generate_cashflows1();
    let date = Local.ymd(2018, 3, 31);
    let _irr = irr(leg, 0.0, dc, Compounding::Compounded, 
                Frequency::Annual, true, date, date, 1.0E-6, 100, 0.05);
    eprintln!("calc irr is {}", _irr);
}

fn irr(
        leg_:Leg,
        npv_:Real,
        dayCounter_:Box<DayCounter>,
        compounding_:Compounding,
        frequency_:Frequency,
        includeSettlementDateFlows_:bool,
        settlementDate_:LocalDate, 
        npvDate_:LocalDate,
        accuracy:Real,
        maxIterations:usize,
        guess:Real)->Real {
        let mut solver = Solver1D::default();
        solver.setMaxEvaluations(maxIterations);
        return irr_(solver, leg_, npv_, dayCounter_,
                    compounding_, frequency_,
                    includeSettlementDateFlows_,
                    settlementDate_, npvDate_,
                    accuracy, guess);
}

// template <typename Solver>
fn irr_(mut solver:Solver1D,
    leg_:Leg,
    npv_:Real,
    dayCounter_:Box<DayCounter>,
    compounding_:Compounding,
    frequency_:Frequency,
    includeSettlementDateFlows_:bool,
    settlementDate_:LocalDate, 
    npvDate_:LocalDate,
    accuracy:Real,
    guess:Real
    )->Real {
    let irrFinder = IrrFinder::new(leg_, npv_, dayCounter_, compounding_,
                            frequency_, includeSettlementDateFlows_,
                            settlementDate_, npvDate_);
    return solver.solve(Box::new(NewtonSafeAlgo), irrFinder, accuracy, guess, guess/10.0);
}


struct Solver1D{
    pub root_:Real, 
    pub xMin_:Real, pub xMax_:Real, 
    pub fxMin_:Real, pub fxMax_:Real,
    pub maxEvaluations_:usize,
    pub evaluationNumber_:usize,
    pub lowerBound_:Real, pub upperBound_:Real,
    pub lowerBoundEnforced_:bool, pub upperBoundEnforced_:bool,
}

impl Default for Solver1D {
    fn default() -> Self {
        Solver1D{
            root_ : 0.1,
            xMin_:0.0, xMax_:10.0,
            fxMin_:0.0, fxMax_:100.0,
            maxEvaluations_:MAX_FUNCTION_EVALUATIONS,
            evaluationNumber_:0,
            lowerBound_:0.0, upperBound_:100.0,
            lowerBoundEnforced_:false,
            upperBoundEnforced_:false,

        }
    }
}

trait SolverAlgo{
    fn impl_solve(&self, solver:&mut Solver1D, f:Box<dyn Derivative>, xAccuracy:Real)->Real;
}

struct NewtonAlgo;

impl SolverAlgo for NewtonAlgo{
    fn impl_solve(&self, solver:&mut Solver1D, f:Box<dyn Derivative>, xAccuracy:Real)->Real{
    /* The implementation of the algorithm was inspired by
        Press, Teukolsky, Vetterling, and Flannery,
        "Numerical Recipes in C", 2nd edition,
        Cambridge University Press
    */
        let mut froot:Real;
        let mut dfroot:Real;
        let mut dx:Real;

        froot = f.call(solver.root_);
        dfroot = f.derivative(solver.root_);
        solver.evaluationNumber_ += 1;

        while solver.evaluationNumber_<= solver.maxEvaluations_ {
            dx = froot/dfroot;
            solver.root_ -= dx;
            // jumped out of brackets, switch to NewtonSafe
            // if (solver.xMin_-solver.root_)*(solver.root_-solver.xMax_) < 0.0 {
            //     NewtonSafe s;
            //     s.setMaxEvaluations(maxEvaluations_-evaluationNumber_);
            //     return s.solve(f, xAccuracy, root_+dx, xMin_, xMax_);
            // }
            if dx.abs() < xAccuracy {
                f.call(solver.root_);
                solver.evaluationNumber_ += 1;
                return solver.root_;
            }
            froot = f.call(solver.root_);
            dfroot = f.derivative(solver.root_);
            solver.evaluationNumber_ += 1;
        }

        panic!("maximum number of function evaluations ({}) exceeded", solver.maxEvaluations_);
    }
}

struct NewtonSafeAlgo;
impl SolverAlgo for NewtonSafeAlgo {
      fn impl_solve(&self, solver:&mut Solver1D, f:Box<dyn Derivative>, xAccuracy:Real)->Real{

        /* The implementation of the algorithm was inspired by
            Press, Teukolsky, Vetterling, and Flannery,
            "Numerical Recipes in C", 2nd edition,
            Cambridge University Press
        */
        let mut froot:Real;
        let mut dfroot:Real;
        let mut dx:Real;
        let mut dxold:Real;
        let mut xh:Real;
        let mut xl:Real;

        // Orient the search so that f(xl) < 0
        if solver.fxMin_ < 0.0 {
            xl = solver.xMin_;
            xh = solver.xMax_;
        } else {
            xh = solver.xMin_;
            xl = solver.xMax_;
        }

        // the "stepsize before last"
        dxold = solver.xMax_- solver.xMin_;
        // it was dxold=std::fabs(xMax_-xMin_); in Numerical Recipes
        // here (xMax_-xMin_ > 0) is verified in the constructor

        // and the last step
        dx = dxold;

        froot = f.call(solver.root_);
        dfroot = f.derivative(solver.root_);
        // QL_REQUIRE(dfroot != Null<Real>(),
        //             "NewtonSafe requires function's derivative");
        solver.evaluationNumber_ += 1;

        while solver.evaluationNumber_<= solver.maxEvaluations_ {
            // Bisect if (out of range || not decreasing fast enough)
            if (((solver.root_-xh)*dfroot-froot)*
                    ((solver.root_-xl)*dfroot-froot) > 0.0)
                || 2.0*froot.abs() > (dxold*dfroot).abs() {
                dxold = dx;
                dx = (xh-xl)/2.0;
                solver.root_=xl+dx;
            } else {
                dxold = dx;
                dx = froot/dfroot;
                solver.root_ -= dx;
            }
            // Convergence criterion
            if dx.abs() < xAccuracy {
                f.call(solver.root_);
                solver.evaluationNumber_ += 1;
                return solver.root_;
            }
            froot = f.call(solver.root_);
            dfroot = f.derivative(solver.root_);
            solver.evaluationNumber_ += 1;
            if froot < 0.0{
                xl = solver.root_;
            } else {
                xh = solver.root_;
            }
        }

        panic!("maximum number of function evaluations ({}) exceeded", solver.maxEvaluations_);
    }
}


impl Solver1D{
    fn setMaxEvaluations(&mut self, evaluations:usize){
        self.maxEvaluations_ = evaluations;
    }
    fn setLowerBound(&mut self, lowerBound:Real){
        self.lowerBound_ = lowerBound;
        self.lowerBoundEnforced_ = true;
    }
    fn setUpperBound(&mut self, upperBound:Real){
        self.upperBound_ = upperBound;
        self.upperBoundEnforced_ = true;
    }

    fn enforceBounds_(&self, x:Real) -> Real {
        if self.lowerBoundEnforced_ && x < self.lowerBound_ {
            return self.lowerBound_
        }
            
        if self.upperBoundEnforced_ && x > self.upperBound_{
            return self.upperBound_;
        }
        return x;
    }

    fn solve(&mut self, solver_algo:Box<dyn SolverAlgo>, f:Box<dyn Derivative>, mut accuracy:Real, guess:Real, step:Real)->Real{

            assert!(accuracy>0.0);
            // check whether we really want to use epsilon
            accuracy = accuracy.max(QL_EPSILON);
          
            let mut flipflop:i32 = -1;

            self.root_ = guess;
            self.fxMax_ = f.call(self.root_);

            // monotonically crescent bias, as in optionValue(volatility)
            if close(self.fxMax_, 0.0){
                return self.root_;
            }else if self.fxMax_ > 0.0 {
                self.xMin_ = self.enforceBounds_(self.root_ - step);
                self.fxMin_ = f.call(self.xMin_);
                self.xMax_ = self.root_;
            } else {
                self.xMin_ = self.root_;
                self.fxMin_ = self.fxMax_;
                self.xMax_ = self.enforceBounds_(self.root_+step);
                self.fxMax_ = f.call(self.xMax_);
            }

            self.evaluationNumber_ = 2;
            while self.evaluationNumber_ <= self.maxEvaluations_ {
                if self.fxMin_*self.fxMax_ <= 0.0 {
                    if close(self.fxMin_, 0.0){
                        return self.xMin_;
                    }
                    if close(self.fxMax_, 0.0){
                        return self.xMax_;
                    }
                    self.root_ = (self.xMax_+self.xMin_)/2.0;
                    return solver_algo.impl_solve(self, f, accuracy);
                }
                if self.fxMin_.abs() < self.fxMax_.abs() {
                    self.xMin_ = self.enforceBounds_(self.xMin_+growthFactor*(self.xMin_ - self.xMax_));
                    self.fxMin_= f.call(self.xMin_);
                } else if self.fxMin_.abs() > self.fxMax_.abs() {
                    self.xMax_ = self.enforceBounds_(self.xMax_+growthFactor*(self.xMax_ - self.xMin_));
                    self.fxMax_= f.call(self.xMax_);
                } else if flipflop == -1 {
                    self.xMin_ = self.enforceBounds_(self.xMin_+growthFactor*(self.xMin_ - self.xMax_));
                    self.fxMin_= f.call(self.xMin_);
                    self.evaluationNumber_ += 1;
                    flipflop = 1;
                } else if flipflop == 1 {
                    self.xMax_ = self.enforceBounds_(self.xMax_+growthFactor*(self.xMax_ - self.xMin_));
                    self.fxMax_= f.call(self.xMax_);
                    flipflop = -1;
                }
                self.evaluationNumber_+=1;
            }

            panic!("unable to bracket root in {} function evaluations (last bracket attempt: f[{},{}] -> [{},{}])",
                self.maxEvaluations_, self.xMin_, self.xMax_, self.fxMin_, self.fxMax_);
    }
}