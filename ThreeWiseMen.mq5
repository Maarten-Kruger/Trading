#property copyright "MJ Kruger"
#property link      "https://github.com/Maarten-Kruger/Trading"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>
#include <Trade/PositionInfo.mqh>

//--- input parameters
input int      InpAvgPeriod        = 5;      // AVGPERIOD for rolling median average
input int      InpBacklog          = 20;     // BACKLOG size
input int      InpNRange           = 2;      // NRANGE for peak/valley detection
input double   InpPercent          = 0.5;    // PERCENT between points
input int      InpTradesChecking   = 10;     // TRADESCHECKING bars
input int      InpATRPeriod        = 14;     // ATR period
input double   InpStopATR1         = 2.0;    // STOPATR1 multiplier
input double   InpStopATR2         = 1.5;    // STOPATR2 multiplier
input double   InpStopATR3         = 1.0;    // STOPATR3 multiplier
input double   InpRR1              = 2.0;    // RR1 risk reward
input double   InpRR2              = 1.5;    // RR2 risk reward
input double   InpRR3              = 1.0;    // RR3 risk reward
input int      InpHalfSLBars       = 5;      // HALFSL bars for stop reduction
input int      InpTCloseBars       = 30;     // TCLOSE bars to close trade
input double   InpRiskPercent      = 1.0;    // Risk per trade (% equity)
input uint     InpSlippage         = 10;     // Slippage in points
input long     InpMagic            = 20250827; // Magic number
//--- optimization weights
input double   InpW_T              = 33.34;  // Weight: trade density
input double   InpW_P              = 33.33;  // Weight: monthly profit
input double   InpW_D              = 33.33;  // Weight: drawdown

//--- trade classes
CTrade  trade;
CPositionInfo pos_info;

//--- data arrays
double   backlog[];          // rolling data array
bool     wave_active = false;
bool     wave_up = false;
int      trade_bars_left = 0;
double   wave_p[9];          // store points 1..8 values (index 0..8)
int      atr_handle=INVALID_HANDLE; // handle for ATR indicator

//--- structure to track open trades
struct TradeInfo
  {
   ulong     ticket;
   double    entry;
   double    initial_sl;
   double    risk;
   int       stage;       // 0 original,1 50%,2 25%,3 breakeven
   datetime  open_time;
  };

TradeInfo trades[];

// helper to remove element from trades array
void RemoveTrade(int index)
  {
   int sz=ArraySize(trades);
   if(index<0 || index>=sz) return;
   for(int j=index;j<sz-1;j++)
      trades[j]=trades[j+1];
   ArrayResize(trades,sz-1);
  }

//--- testing period info
datetime test_start=0;
datetime test_end=0;
double   start_equity=0.0;

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
  {
   ArrayResize(backlog, InpBacklog);
   ArrayInitialize(backlog, 0.0);

   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(InpSlippage);

   start_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(Bars(_Symbol, _Period) > 0)
      test_start = iTime(_Symbol, _Period, Bars(_Symbol, _Period)-1);
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
//| Calculate ATR                                                     |
//+------------------------------------------------------------------+
double GetATR()
  {
   if(atr_handle==INVALID_HANDLE)
      atr_handle=iATR(_Symbol,_Period,InpATRPeriod);
   double atr_buf[];
   if(CopyBuffer(atr_handle,0,1,1,atr_buf)<=0)
      return(0.0);
   return(atr_buf[0]);
  }

//+------------------------------------------------------------------+
//| Helper: detect new bar                                            |
//+------------------------------------------------------------------+
bool IsNewBar()
  {
   static datetime last_bar=0;
   datetime cur_bar=iTime(_Symbol,_Period,0);
   if(cur_bar!=last_bar)
     {
      last_bar=cur_bar;
      return(true);
     }
   return(false);
  }

//+------------------------------------------------------------------+
//| Helper: shift backlog                                             |
//+------------------------------------------------------------------+
void AppendBacklog(double value)
  {
   for(int i=0;i<InpBacklog-1;i++)
      backlog[i]=backlog[i+1];
   backlog[InpBacklog-1]=value;
  }

//+------------------------------------------------------------------+
//| Helper: peak/valley detection                                     |
//+------------------------------------------------------------------+
bool IsPeak(int index)
  {
   double val=backlog[index];
   for(int i=index-InpNRange;i<=index+InpNRange;i++)
     {
      if(i<0 || i>=ArraySize(backlog) || i==index) continue;
      if(backlog[i]>val) return(false);
     }
   return(true);
  }

bool IsValley(int index)
  {
   double val=backlog[index];
   for(int i=index-InpNRange;i<=index+InpNRange;i++)
     {
      if(i<0 || i>=ArraySize(backlog) || i==index) continue;
      if(backlog[i]<val) return(false);
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| Wave detection                                                    |
//+------------------------------------------------------------------+
bool DetectWave()
  {
   int last=ArraySize(backlog)-1;
   if(last<3) return(false);

   double p4=backlog[last];
   double maxv=backlog[0],minv=backlog[0];
   for(int i=1;i<last;i++)
     {
      if(backlog[i]>maxv) maxv=backlog[i];
      if(backlog[i]<minv) minv=backlog[i];
     }
   double extreme_tol=(maxv-minv)*0.1;

   if(p4>=maxv-extreme_tol) // potential up wave
     {
      for(int p2=1;p2<last-1;p2++)
        {
         if(!IsPeak(p2)) continue;
         // find point1 before p2 that is valley
         int p1=-1;
         double p1val=DBL_MAX;
         for(int i=0;i<p2;i++)
           {
            if(IsValley(i) && backlog[i]<p1val)
              {
               p1=i; p1val=backlog[i];
              }
           }
         if(p1==-1) continue;
         // find point3 after p2 that is valley
         for(int p3=p2+1;p3<last;p3++)
           {
            if(!IsValley(p3)) continue;
            double expected=p1val + InpPercent*(backlog[p2]-p1val);
            double tol=MathMax(_Point*10,MathAbs(backlog[p2]-p1val)*0.1);
            if(MathAbs(backlog[p3]-expected) <= tol)
              {
               // valid wave found
               wave_p[0]=backlog[p1];
               wave_p[1]=backlog[p2];
               wave_p[2]=backlog[p3];
               wave_p[3]=p4;
               wave_up=true;
               wave_active=true;
               trade_bars_left=InpTradesChecking;
               return(true);
              }
           }
        }
     }
   else if(p4<=minv+extreme_tol) // potential down wave
     {
      for(int p2=1;p2<last-1;p2++)
        {
         if(!IsValley(p2)) continue;
         // find point1 before p2 that is peak
         int p1=-1; double p1val=-DBL_MAX;
         for(int i=0;i<p2;i++)
           {
            if(IsPeak(i) && backlog[i]>p1val)
              {
               p1=i; p1val=backlog[i];
              }
           }
         if(p1==-1) continue;
         for(int p3=p2+1;p3<last;p3++)
           {
            if(!IsPeak(p3)) continue;
            double expected=p1val + InpPercent*(backlog[p2]-p1val);
            double tol=MathMax(_Point*10,MathAbs(backlog[p2]-p1val)*0.1);
            if(MathAbs(backlog[p3]-expected) <= tol)
              {
               wave_p[0]=backlog[p1];
               wave_p[1]=backlog[p2];
               wave_p[2]=backlog[p3];
               wave_p[3]=p4;
               wave_up=false;
               wave_active=true;
               trade_bars_left=InpTradesChecking;
               return(true);
              }
           }
        }
     }
   return(false);
  }

//+------------------------------------------------------------------+
//| Trade placement                                                   |
//+------------------------------------------------------------------+
void CheckTrades()
  {
   if(!wave_active) return;
   double atr=GetATR();
   double high1=iHigh(_Symbol,_Period,1);
   double low1=iLow(_Symbol,_Period,1);

   if(wave_up)
     {
      // track highest high after point 4 without requiring a confirmed peak
      if(wave_p[4]==0.0 || high1>wave_p[4])
         wave_p[4]=high1;

      // point 6 is a retrace level between point 3 and the running point 5
      wave_p[5]=wave_p[2] + InpPercent*(wave_p[4]-wave_p[2]);
      if(low1<=wave_p[5] && wave_p[6]==0.0)
        {
         double sl=wave_p[2] - InpStopATR1*atr;
         double risk=wave_p[5]-sl;
         double tp=wave_p[5]+risk*InpRR1;
         int sl_points=(int)MathMax(1,MathRound(risk/_Point));
         double vol=CalcVolumeByRisk(InpRiskPercent,sl_points);
         if(vol>0 && trade.Buy(vol,_Symbol,wave_p[5],sl,tp,"WM1"))
           {
            TradeInfo ti; ti.ticket=trade.ResultOrder(); ti.entry=wave_p[5]; ti.initial_sl=sl; ti.risk=risk; ti.stage=0; ti.open_time=TimeCurrent();
            int sz=ArraySize(trades); ArrayResize(trades,sz+1); trades[sz]=ti;
            wave_p[6]=wave_p[5];
           }
        }
      if(wave_p[6]>0.0 && high1>wave_p[6] && wave_p[7]==0.0)
        {
         double entry=high1;
         double sl=wave_p[2]-InpStopATR2*atr;
         double risk=entry-sl;
         double tp=entry+risk*InpRR2;
         int sl_points=(int)MathMax(1,MathRound(risk/_Point));
         double vol=CalcVolumeByRisk(InpRiskPercent,sl_points);
         if(vol>0 && trade.Buy(vol,_Symbol,entry,sl,tp,"WM2"))
           {
            TradeInfo ti; ti.ticket=trade.ResultOrder(); ti.entry=entry; ti.initial_sl=sl; ti.risk=risk; ti.stage=0; ti.open_time=TimeCurrent();
            int sz=ArraySize(trades); ArrayResize(trades,sz+1); trades[sz]=ti;
            wave_p[7]=entry;
           }
        }
      if(wave_p[7]>0.0 && high1>=wave_p[4] && wave_p[8]==0.0)
        {
         double entry=high1;
         double sl=wave_p[2]-InpStopATR3*atr;
         double risk=entry-sl;
         double tp=entry+risk*InpRR3;
         int sl_points=(int)MathMax(1,MathRound(risk/_Point));
         double vol=CalcVolumeByRisk(InpRiskPercent,sl_points);
         if(vol>0 && trade.Buy(vol,_Symbol,entry,sl,tp,"WM3"))
           {
            TradeInfo ti; ti.ticket=trade.ResultOrder(); ti.entry=entry; ti.initial_sl=sl; ti.risk=risk; ti.stage=0; ti.open_time=TimeCurrent();
            int sz=ArraySize(trades); ArrayResize(trades,sz+1); trades[sz]=ti;
            wave_p[8]=entry;
           }
        }
     }
   else
     {
      // track lowest low after point 4 without requiring a confirmed valley
      if(wave_p[4]==0.0 || low1<wave_p[4])
         wave_p[4]=low1;

      wave_p[5]=wave_p[2] + InpPercent*(wave_p[4]-wave_p[2]);
      if(high1>=wave_p[5] && wave_p[6]==0.0)
        {
         double sl=wave_p[2]+InpStopATR1*atr;
         double risk=sl-wave_p[5];
         double tp=wave_p[5]-risk*InpRR1;
         int sl_points=(int)MathMax(1,MathRound(risk/_Point));
         double vol=CalcVolumeByRisk(InpRiskPercent,sl_points);
         if(vol>0 && trade.Sell(vol,_Symbol,wave_p[5],sl,tp,"WM1"))
           {
            TradeInfo ti; ti.ticket=trade.ResultOrder(); ti.entry=wave_p[5]; ti.initial_sl=sl; ti.risk=risk; ti.stage=0; ti.open_time=TimeCurrent();
            int sz=ArraySize(trades); ArrayResize(trades,sz+1); trades[sz]=ti;
            wave_p[6]=wave_p[5];
           }
        }
      if(wave_p[6]>0.0 && low1<wave_p[6] && wave_p[7]==0.0)
        {
         double entry=low1;
         double sl=wave_p[2]+InpStopATR2*atr;
         double risk=sl-entry;
         double tp=entry-risk*InpRR2;
         int sl_points=(int)MathMax(1,MathRound(risk/_Point));
         double vol=CalcVolumeByRisk(InpRiskPercent,sl_points);
         if(vol>0 && trade.Sell(vol,_Symbol,entry,sl,tp,"WM2"))
           {
            TradeInfo ti; ti.ticket=trade.ResultOrder(); ti.entry=entry; ti.initial_sl=sl; ti.risk=risk; ti.stage=0; ti.open_time=TimeCurrent();
            int sz=ArraySize(trades); ArrayResize(trades,sz+1); trades[sz]=ti;
            wave_p[7]=entry;
           }
        }
      if(wave_p[7]>0.0 && low1<=wave_p[4] && wave_p[8]==0.0)
        {
         double entry=low1;
         double sl=wave_p[2]+InpStopATR3*atr;
         double risk=sl-entry;
         double tp=entry-risk*InpRR3;
         int sl_points=(int)MathMax(1,MathRound(risk/_Point));
         double vol=CalcVolumeByRisk(InpRiskPercent,sl_points);
         if(vol>0 && trade.Sell(vol,_Symbol,entry,sl,tp,"WM3"))
           {
            TradeInfo ti; ti.ticket=trade.ResultOrder(); ti.entry=entry; ti.initial_sl=sl; ti.risk=risk; ti.stage=0; ti.open_time=TimeCurrent();
            int sz=ArraySize(trades); ArrayResize(trades,sz+1); trades[sz]=ti;
            wave_p[8]=entry;
           }
        }
     }

   trade_bars_left--;
   if(trade_bars_left<=0 || (wave_p[6]>0 && wave_p[7]>0 && wave_p[8]>0))
     {
      wave_active=false;
      ArrayInitialize(wave_p,0.0);
     }
  }

//+------------------------------------------------------------------+
//| Manage open trades                                                |
//+------------------------------------------------------------------+
void ManageTrades()
  {
   for(int i=ArraySize(trades)-1;i>=0;i--)
     {
      TradeInfo ti=trades[i];
      if(!pos_info.SelectByTicket(ti.ticket))
        {
         RemoveTrade(i);
         continue;
        }
      int bars_elapsed=BarsSince(ti.open_time);
      double entry=ti.entry;
      double sl=ti.initial_sl;
      double risk=ti.risk;
      double new_sl=0.0;
      if(ti.stage==0 && bars_elapsed>=InpHalfSLBars)
        {
         if(pos_info.PositionType()==POSITION_TYPE_BUY)
            new_sl=entry-risk*0.5;
         else
            new_sl=entry+risk*0.5;
         ti.stage=1;
        }
      else if(ti.stage==1 && bars_elapsed>=InpHalfSLBars*2)
        {
         if(pos_info.PositionType()==POSITION_TYPE_BUY)
            new_sl=entry-risk*0.25;
         else
            new_sl=entry+risk*0.25;
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
         RemoveTrade(i);
         continue;
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
//| Helper: bars since time                                          |
//+------------------------------------------------------------------+
int BarsSince(datetime t)
  {
   int idx=iBarShift(_Symbol,_Period,t,false);
   if(idx<0) return(0);
   return(idx);
  }

//+------------------------------------------------------------------+
//| Calculate volume by risk                                          |
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
   double sl_price=sl_points*_Point;
   double loss_per_lot=sl_price*money_per_price_unit;
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
//| OnTick                                                            |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(!IsNewBar()) return;

   // fill backlog
   double median_sum=0.0;
   for(int i=0;i<InpAvgPeriod;i++)
      median_sum += (iHigh(_Symbol,_Period,i)+iLow(_Symbol,_Period,i))/2.0;
   double avg=median_sum/InpAvgPeriod;
   AppendBacklog(avg);
   test_end=iTime(_Symbol,_Period,0);

   if(!wave_active)
      DetectWave();
   else
      CheckTrades();

   ManageTrades();
  }

//+------------------------------------------------------------------+
//| Helper: months difference                                         |
//+------------------------------------------------------------------+
int MonthsDiff(datetime from_time,datetime to_time)
  {
   MqlDateTime a,b;
   TimeToStruct(from_time,a);
   TimeToStruct(to_time,b);
   int months=(b.year-a.year)*12+(b.mon-a.mon);
   if(b.day<a.day) months-=1;
   if(months<1) months=1;
   return(months);
  }

//+------------------------------------------------------------------+
//| OnTester - custom optimization criterion                           |
//+------------------------------------------------------------------+
double OnTester()
  {
   int bars_total=Bars(_Symbol,_Period);
   if(bars_total<=0) return(0.0);
   int months=MonthsDiff(test_start,test_end);
   double trades_total=TesterStatistics(STAT_TRADES);
   double density=(bars_total>0? trades_total/bars_total:0.0);
   double profit=TesterStatistics(STAT_PROFIT); // total profit in currency
   double monthly_profit=(start_equity>0? (profit/start_equity)/months:0.0);
   double dd=TesterStatistics(STAT_EQUITY_DDREL_PERCENT)/100.0;
   double score=density*InpW_T + monthly_profit*InpW_P - dd*InpW_D;
   return(score*100.0);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
