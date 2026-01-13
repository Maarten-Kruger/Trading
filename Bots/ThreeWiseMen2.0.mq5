#property copyright "MJ Kruger"
#property link      "https://github.com/Maarten-Kruger/Trading"
#property version   "2.0"
#property strict

#include <Trade/Trade.mqh>
#include <Trade/PositionInfo.mqh>

//--- input parameters for data collection
input int      InpAvgPeriod      = 5;      // N: candles used for rolling median average
input int      InpBacklog        = 40;     // BACKLOG size

//--- example wave generation
input double   InpScale          = 1.0;    // S: scaling factor
input double   InpB              = 1.0;    // B: extremity factor in f(x)
input double   InpB1             = -1.0;   // B1: lower boundary for x
input double   InpB2             = 1.0;    // B2: upper boundary for x
input int      InpWaveLen        = 20;     // N: length of the example wave
input double   InpPearson        = 0.8;    // PEARSON threshold
input double   InpMAE            = 5.0;    // MAE threshold in points

//--- trade checking and wise men parameters
input int      InpTradesChecking = 15;     // TRADESCHECKING bars
input double   InpPercent        = 0.5;    // PERCENT between point5 and point3
input int      InpATRPeriod      = 14;     // ATR period
input double   InpStopATR1       = 2.0;    // STOPATR1 multiplier
input double   InpStopATR2       = 1.5;    // STOPATR2 multiplier
input double   InpStopATR3       = 1.0;    // STOPATR3 multiplier
input double   InpRR1            = 2.0;    // RR1
input double   InpRR2            = 1.5;    // RR2
input double   InpRR3            = 1.0;    // RR3
input int      InpHalfSLBars     = 5;      // HALFSL intervals
input int      InpTCloseBars     = 30;     // TCLOSE intervals
input double   InpRiskPercent    = 1.0;    // risk per trade (% of equity)
input uint     InpSlippage       = 10;     // slippage in points
input long     InpMagic          = 20250827; // magic number

//--- optimization weights (Wt + Wp + Wd = 100)
input double   InpW_T            = 33.34;  // trade density weight
input double   InpW_P            = 33.33;  // monthly profit weight
input double   InpW_D            = 33.33;  // drawdown weight

//--- trade and market access
CTrade        trade;
CPositionInfo pos_info;

//--- data arrays and state
double   backlog[];            // rolling median prices
double   example_wave[];       // precomputed example wave
bool     wave_active=false;    // currently in trade checking window
bool     wave_up=false;        // direction of detected wave
int      trade_bars_left=0;    // remaining bars to search for trades
int      backlog_filled=0;     // number of valid entries in backlog

double   point3=0.0, point5=0.0;   // points from strategy description
bool     wm1=false, wm2=false, wm3=false; // trades placed flags

//--- indicator handle
int      atr_handle=INVALID_HANDLE;

//--- trade tracking structure
struct TradeInfo
  {
   ulong     ticket;       // position ticket
   double    entry;        // entry price
   double    initial_sl;   // initial stop loss
   double    risk;         // initial risk per lot
   int       stage;        // 0 original,1=50%,2=25%,3=breakeven
   datetime  open_time;    // open time
  };
TradeInfo trades[];

//--- testing period variables
datetime test_start=0;      // beginning of test
datetime test_end=0;        // end of test

double   start_equity=0.0; // used for monthly profit calculation

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
  {
   ArrayResize(backlog,InpBacklog);
   ArrayInitialize(backlog,0.0);
   backlog_filled=0;

   // pre-generate example wave once at start
   GenerateExample(example_wave);

   // print the example wave once at initialization
   string examp="Example Array: ";
   for(int i=0;i<ArraySize(example_wave);i++)
     {
      examp+=DoubleToString(example_wave[i],6)+" ";
     }
   Print(examp);

   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(InpSlippage);

   start_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(Bars(_Symbol,_Period)>0)
      test_start=iTime(_Symbol,_Period,Bars(_Symbol,_Period)-1);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(atr_handle!=INVALID_HANDLE)
      IndicatorRelease(atr_handle);
  }

//+------------------------------------------------------------------+
//| Helper: obtain ATR value                                         |
//+------------------------------------------------------------------+
double GetATR()
  {
   if(atr_handle==INVALID_HANDLE)
      atr_handle=iATR(_Symbol,_Period,InpATRPeriod);
   double buf[];
   if(CopyBuffer(atr_handle,0,1,1,buf)<=0)
      return(0.0);
   return(buf[0]);
  }

//+------------------------------------------------------------------+
//| Helper: detect new bar                                           |
//+------------------------------------------------------------------+
bool IsNewBar()
  {
   static datetime last=0;
   datetime t=iTime(_Symbol,_Period,0);
   if(t!=last)
     {
      last=t; return(true);
     }
   return(false);
  }

//+------------------------------------------------------------------+
//| Append value to backlog                                          |
//+------------------------------------------------------------------+
void AppendBacklog(double v)
  {
   for(int i=0;i<InpBacklog-1;i++)
      backlog[i]=backlog[i+1];
   backlog[InpBacklog-1]=v;
   if(backlog_filled<InpBacklog)
      backlog_filled++;
  }

//+------------------------------------------------------------------+
//| Generate example wave array                                      |
//+------------------------------------------------------------------+
void GenerateExample(double &example[])
  {
   ArrayResize(example,InpWaveLen);
   double dx=(InpB2-InpB1)/(InpWaveLen-1);
   for(int i=0;i<InpWaveLen;i++)
     {
      double x=InpB1+dx*i;
      example[i]=InpScale*(x*x*x-InpB*x);
     }
  }

//+------------------------------------------------------------------+
//| Detect wave using correlation & MAE                              |
//+------------------------------------------------------------------+
bool DetectWave()
  {
   // ensure backlog holds enough valid data and can supply wave length
   if(backlog_filled<InpWaveLen || InpWaveLen>ArraySize(backlog))
      return(false);

   // build arrays B and E of identical size
   double B[]; ArrayResize(B,InpWaveLen);
   for(int i=0;i<InpWaveLen;i++)
      B[i]=backlog[ArraySize(backlog)-InpWaveLen+i];

   double E[]; GenerateExample(E);
   if(ArraySize(E)!=ArraySize(B))
      return(false);
   double offset=B[0]-E[0];
   for(int i=0;i<InpWaveLen;i++)
      E[i]+=offset; // align start levels precisely

   string backt="BackLog Array: ";
   for(int i=0;i<ArraySize(B);i++)
     {
      backt+=DoubleToString(B[i],6)+" ";
     }
   Print(backt);
   
   string examp="Example Array: ";
   for(int i=0;i<ArraySize(E);i++)
     {
      examp+=DoubleToString(E[i],6)+" ";
     }
   Print(examp);


   // Pearson correlation
   double meanB=0.0,meanE=0.0;
   for(int i=0;i<InpWaveLen;i++) { meanB+=B[i]; meanE+=E[i]; }
   meanB/=InpWaveLen; meanE/=InpWaveLen;
   double num=0.0,denB=0.0,denE=0.0;
   for(int i=0;i<InpWaveLen;i++)
     {
      double db=B[i]-meanB;
      double de=E[i]-meanE;
      num+=db*de; denB+=db*db; denE+=de*de;
     }
   double denom=MathSqrt(denB*denE);
   if(denom==0.0) return(false);
   double p=num/denom;
   
   PrintFormat("Value of P is: %f", p);

   // MAE
   double sum=0.0;
   for(int i=1;i<InpWaveLen;i++)
     {
      sum+=MathAbs(E[i]-B[i])/_Point; // in points
     }
   double d=sum/(InpWaveLen-1);

   PrintFormat("Value of D is: %f", d);

   // thresholds: PEARSON < |p| and MAE > |d|
   if(MathAbs(p) > InpPearson && d < InpMAE)
     {
      wave_up=(p>0);
      // find point3 in example (local min or max)
      int pivot=-1;
      for(int i=1;i<InpWaveLen-1;i++)
        {
         if(wave_up)
           {
            if(E[i-1]>E[i] && E[i]<E[i+1]) { pivot=i; break; }
           }
         else
           {
            if(E[i-1]<E[i] && E[i]>E[i+1]) { pivot=i; break; }
           }
        }
      if(pivot==-1) return(false);
      point3=B[pivot];
      point5=point3; // initialize point5 tracker
      wm1=wm2=wm3=false;
      trade_bars_left=InpTradesChecking;
      wave_active=true;
      return(true);
     }
   return(false);
  }

//+------------------------------------------------------------------+
//| Place wise man trade                                              |
//+------------------------------------------------------------------+
bool PlaceTrade(int wm,ENUM_ORDER_TYPE type,double stop_mult,double rr)
  {
   double atr=GetATR();
   if(atr<=0.0) return(false);
   double entry=0.0,sl=0.0,tp=0.0;
   if(type==ORDER_TYPE_BUY)
     {
      SymbolInfoDouble(_Symbol,SYMBOL_ASK,entry);
      sl=point3-atr*stop_mult;
      double risk=entry-sl;
      tp=entry+risk*rr;
      int sl_pts=(int)MathRound(risk/_Point);
      double vol=CalcVolumeByRisk(InpRiskPercent,sl_pts);
      if(vol<=0.0) return(false);
      if(!trade.Buy(vol,_Symbol,0.0,sl,tp,"WM"+IntegerToString(wm))) return(false);
      TradeInfo ti; ti.ticket=trade.ResultOrder(); ti.entry=entry; ti.initial_sl=sl; ti.risk=risk; ti.stage=0; ti.open_time=TimeCurrent();
      int sz=ArraySize(trades); ArrayResize(trades,sz+1); trades[sz]=ti;
      return(true);
     }
   else
     {
      SymbolInfoDouble(_Symbol,SYMBOL_BID,entry);
      sl=point3+atr*stop_mult;
      double risk=sl-entry;
      tp=entry-risk*rr;
      int sl_pts=(int)MathRound(risk/_Point);
      double vol=CalcVolumeByRisk(InpRiskPercent,sl_pts);
      if(vol<=0.0) return(false);
      if(!trade.Sell(vol,_Symbol,0.0,sl,tp,"WM"+IntegerToString(wm))) return(false);
      TradeInfo ti; ti.ticket=trade.ResultOrder(); ti.entry=entry; ti.initial_sl=sl; ti.risk=risk; ti.stage=0; ti.open_time=TimeCurrent();
      int sz=ArraySize(trades); ArrayResize(trades,sz+1); trades[sz]=ti;
      return(true);
     }
  }

//+------------------------------------------------------------------+
//| Check for wise men trade setups                                   |
//+------------------------------------------------------------------+
void CheckTrades()
  {
   if(!wave_active) return;
   trade_bars_left--;
   double price=backlog[ArraySize(backlog)-1];

   if(wave_up)
     {
      if(price>point5) point5=price; // track highest point
      double level6=point3 + (point5-point3)*InpPercent;
      if(!wm1 && price<=level6)
        {
         if(PlaceTrade(1,ORDER_TYPE_BUY,InpStopATR1,InpRR1))
            wm1=true;
        }
      else if(wm1 && !wm2 && price>level6)
        {
         if(PlaceTrade(2,ORDER_TYPE_BUY,InpStopATR2,InpRR2))
            wm2=true;
        }
      else if(wm2 && !wm3 && price>=point5)
        {
         if(PlaceTrade(3,ORDER_TYPE_BUY,InpStopATR3,InpRR3))
            wm3=true;
        }
     }
   else // down wave
     {
      if(price<point5) point5=price; // track lowest point
      double level6=point3 - (point3-point5)*InpPercent;
      if(!wm1 && price>=level6)
        {
         if(PlaceTrade(1,ORDER_TYPE_SELL,InpStopATR1,InpRR1))
            wm1=true;
        }
      else if(wm1 && !wm2 && price<level6)
        {
         if(PlaceTrade(2,ORDER_TYPE_SELL,InpStopATR2,InpRR2))
            wm2=true;
        }
      else if(wm2 && !wm3 && price<=point5)
        {
         if(PlaceTrade(3,ORDER_TYPE_SELL,InpStopATR3,InpRR3))
            wm3=true;
        }
     }

   if(trade_bars_left<=0)
      wave_active=false;
  }

//+------------------------------------------------------------------+
//| Manage open trades: progressive stop movement and time close     |
//+------------------------------------------------------------------+
void ManageTrades()
  {
   for(int i=ArraySize(trades)-1;i>=0;i--)
     {
      TradeInfo ti=trades[i];
      if(!pos_info.Select(ti.ticket))
        {
         RemoveTrade(i); continue;
        }
      double entry=ti.entry;
      double risk=ti.risk;
      datetime opentime=ti.open_time;
      int bars_elapsed=BarsSince(opentime);
      double new_sl=0.0;
      if(ti.stage==0 && bars_elapsed>=InpHalfSLBars)
        {
         if(pos_info.PositionType()==POSITION_TYPE_BUY) new_sl=entry-risk*0.5;
         else new_sl=entry+risk*0.5;
         ti.stage=1;
        }
      else if(ti.stage==1 && bars_elapsed>=InpHalfSLBars*2)
        {
         if(pos_info.PositionType()==POSITION_TYPE_BUY) new_sl=entry-risk*0.25;
         else new_sl=entry+risk*0.25;
         ti.stage=2;
        }
      else if(ti.stage==2 && bars_elapsed>=InpHalfSLBars*3)
        {
         new_sl=entry;
         ti.stage=3;
        }
      else if(bars_elapsed>=InpTCloseBars)
        {
         trade.PositionClose(ti.ticket);
         RemoveTrade(i); continue;
        }
      if(new_sl>0.0)
        {
         if(pos_info.PositionType()==POSITION_TYPE_BUY && new_sl>pos_info.StopLoss())
            trade.PositionModify(ti.ticket,new_sl,pos_info.TakeProfit());
         else if(pos_info.PositionType()==POSITION_TYPE_SELL && new_sl<pos_info.StopLoss())
            trade.PositionModify(ti.ticket,new_sl,pos_info.TakeProfit());
        }
      trades[i]=ti;
     }
  }

//+------------------------------------------------------------------+
//| Remove trade from tracking array                                  |
//+------------------------------------------------------------------+
void RemoveTrade(int index)
  {
   int sz=ArraySize(trades);
   if(index<0 || index>=sz) return;
   for(int j=index;j<sz-1;j++) trades[j]=trades[j+1];
   ArrayResize(trades,sz-1);
  }

//+------------------------------------------------------------------+
//| Bars since a given time                                           |
//+------------------------------------------------------------------+
int BarsSince(datetime t)
  {
   int idx=iBarShift(_Symbol,_Period,t,false);
   if(idx<0) return(0);
   return(idx);
  }

//+------------------------------------------------------------------+
//| Volume calculation helper                                         |
//+------------------------------------------------------------------+
double CalcVolumeByRisk(double risk_percent,int sl_points)
  {
   if(risk_percent<=0.0 || sl_points<=0) return(0.0);
   double equity=AccountInfoDouble(ACCOUNT_EQUITY);
   double risk_money=equity*(risk_percent/100.0);
   double tick_value=0.0,tick_size=0.0;
   if(!SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_VALUE,tick_value)) return(0.0);
   if(!SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE,tick_size)) return(0.0);
   double money_per_price_unit=(tick_size>0.0?tick_value/tick_size:0.0);
   if(money_per_price_unit<=0.0) return(0.0);
   double loss_per_lot=sl_points*_Point*money_per_price_unit;
   if(loss_per_lot<=0.0) return(0.0);
   double vol=risk_money/loss_per_lot;
   double min_lot=0,max_lot=0,step=0;
   SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN,min_lot);
   SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MAX,max_lot);
   SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP,step);
   if(vol<min_lot) vol=min_lot;
   if(vol>max_lot) vol=max_lot;
   if(step>0.0) vol=MathFloor(vol/step)*step;
   int digits=(step>0? (int)MathRound(-MathLog10(step)) : 2);
   vol=NormalizeDouble(vol,digits);
   return(vol);
  }

//+------------------------------------------------------------------+
//| Helper: months difference                                         |
//+------------------------------------------------------------------+
int MonthsDiff(datetime from_time,datetime to_time)
  {
   MqlDateTime a,b; TimeToStruct(from_time,a); TimeToStruct(to_time,b);
   int months=(b.year-a.year)*12+(b.mon-a.mon);
   if(b.day<a.day) months-=1; if(months<1) months=1; return(months);
  }

//+------------------------------------------------------------------+
//| Get bars and months in testing period                             |
//+------------------------------------------------------------------+
void GetBarsAndMonths(int &total_bars,int &months)
  {
   total_bars=Bars(_Symbol,_Period);
   months=MonthsDiff(test_start,test_end);
  }

//+------------------------------------------------------------------+
//| OnTick                                                            |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(!IsNewBar()) return;

   // update backlog with median price average over InpAvgPeriod
   double med_sum=0.0;
   for(int i=1;i<=InpAvgPeriod;i++)
      med_sum+=(iHigh(_Symbol,_Period,i)+iLow(_Symbol,_Period,i))/2.0;
   double avg=med_sum/InpAvgPeriod;
   AppendBacklog(avg);
   test_end=iTime(_Symbol,_Period,0);

   if(!wave_active)
      DetectWave();
   else
      CheckTrades();

   ManageTrades();
  }

//+------------------------------------------------------------------+
//| OnTester: custom optimization criterion                           |
//+------------------------------------------------------------------+
double OnTester()
  {
   int total_bars=0,months=1; GetBarsAndMonths(total_bars,months);
   double trades_total=TesterStatistics(STAT_TRADES);
   double T=(total_bars>0?trades_total/(double)total_bars:0.0);
   double total_profit=TesterStatistics(STAT_PROFIT);
   double P=(start_equity>0.0 && months>0? (total_profit/start_equity)/months : 0.0);
   double D=TesterStatistics(STAT_EQUITY_DDREL_PERCENT)/100.0; // convert to fraction
   double wt=InpW_T/100.0, wp=InpW_P/100.0, wd=InpW_D/100.0;
   double y=(T*wt + P*wp - D*wd)*100.0;
   return(y);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
