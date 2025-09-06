//+------------------------------------------------------------------+
//|                                            ThreeWiseMen.mq5      |
//|   Implementation of "Three Wise Men" strategy                    |
//|   Strategy based on wave detection and sequential trade setups.  |
//+------------------------------------------------------------------+

#property copyright "OpenAI"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- input parameters
input int      AVGPERIOD      = 5;     // averaging period for median price
input int      NRANGE         = 3;     // range for peak/valley detection
input int      BACKTIME       = 50;    // bars to look back for a wave
input double   PERCENT        = 0.5;   // percent for geometric calculations
input int      TRADESCHECKING = 40;    // bars to check trades after wave
input double   RiskPercent    = 1.0;   // risk per trade (% of equity)
input int      ATRPeriod      = 14;    // ATR period
input double   STOPATR1       = 1.5;   // ATR multiplier for WM1
input double   STOPATR2       = 1.5;   // ATR multiplier for WM2
input double   STOPATR3       = 1.5;   // ATR multiplier for WM3
input double   RR1            = 1.5;   // risk reward WM1
input double   RR2            = 2.0;   // risk reward WM2
input double   RR3            = 2.5;   // risk reward WM3
input int      HALFSL         = 10;    // bars per SL reduction step
input int      TCLOSE         = 100;   // bars to force close trade

//--- optimization weights
input double   Wt             = 40.0;  // trade density weight
input double   Wp             = 40.0;  // profit weight
input double   Wd             = 20.0;  // drawdown weight

//--- trade object
CTrade trade;

//--- magic number
input ulong    Magic          = 12345;

//--- structure to track open trades for management
struct TradeInfo
  {
   ulong   ticket;       // position ticket
   double  originalSL;   // initial stop level
   int     openBar;      // bar index when opened
   int     stage;        // trailing stage
  };
TradeInfo g_trades[];    // dynamic array of active trades

//--- structure for wave state
struct WaveInfo
  {
   bool    active;
   bool    directionUp;
   int     bar4;
   double  p1,p2,p3,p4,p5,p6;
   bool    trade1,trade2,trade3;
  };
WaveInfo g_wave;

//--- start time for optimization statistics
datetime test_start;
int atr_handle;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber(Magic);
   test_start = TimeCurrent();
   g_wave.active=false;
   ArrayResize(g_trades,0);
   atr_handle=iATR(_Symbol,_Period,ATRPeriod); // prepare ATR handle
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   ArrayResize(g_trades,0);
   if(atr_handle!=INVALID_HANDLE)
      IndicatorRelease(atr_handle);
  }

//+------------------------------------------------------------------+
//| Get averaged median price                                        |
//+------------------------------------------------------------------+
double GetAvgPrice(int shift)
  {
   double total=0.0;
   for(int i=0;i<AVGPERIOD;i++)
     {
      // Access price data using iHigh/iLow since global arrays are
      // not directly available in Expert Advisors
      double hi=iHigh(_Symbol,_Period,shift+i);
      double lo=iLow(_Symbol,_Period,shift+i);
      total += (hi+lo)*0.5;
     }
   return(total/AVGPERIOD);
  }

//+------------------------------------------------------------------+
//| Determine if given point is peak                                 |
//+------------------------------------------------------------------+
bool IsPeak(int index)
  {
   double val=GetAvgPrice(index);
   for(int i=1;i<=NRANGE;i++)
     {
      if(GetAvgPrice(index+i)>=val || GetAvgPrice(index-i)>=val)
         return(false);
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| Determine if given point is valley                               |
//+------------------------------------------------------------------+
bool IsValley(int index)
  {
   double val=GetAvgPrice(index);
   for(int i=1;i<=NRANGE;i++)
     {
      if(GetAvgPrice(index+i)<=val || GetAvgPrice(index-i)<=val)
         return(false);
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| Detect wave in last BACKTIME bars                                |
//+------------------------------------------------------------------+
bool DetectWave(WaveInfo &wave)
  {
   if(Bars(_Symbol,_Period)<BACKTIME+NRANGE)
      return(false);

   int start=BACKTIME-1;
   int end=0;

   double p1=GetAvgPrice(start);
   double p4=GetAvgPrice(end);

   bool up = (p4>p1);

   // find P2 (peak) and P3 (valley)
   int idx2=start-1, idx3=start-1;
   double best2=GetAvgPrice(idx2);
   double best3=best2;
   for(int i=start-1;i>NRANGE;i--)
     {
      double price=GetAvgPrice(i);
      if(up)
        {
         if(price>best2){best2=price; idx2=i;}
        }
      else
        {
         if(price<best2){best2=price; idx2=i;}
        }
     }

   // find valley after idx2
   best3=GetAvgPrice(idx2-1);
   idx3=idx2-1;
   for(int i=idx2-1;i>NRANGE;i--)
     {
      double price=GetAvgPrice(i);
      if(up)
        {
         if(price<best3){best3=price; idx3=i;}
        }
      else
        {
         if(price>best3){best3=price; idx3=i;}
        }
     }

   // check conditions
   if(up)
     {
      if(p1>best3 || p4<best2) return(false);
      if(!IsPeak(idx2) || !IsValley(idx3)) return(false);
      double target = p1 + PERCENT*(best2-p1);
      if(MathAbs(best3 - target) > (best2-p1)*0.05) return(false); // 5% tolerance
     }
   else
     {
      if(p1<best3 || p4>best2) return(false);
      if(!IsValley(idx2) || !IsPeak(idx3)) return(false);
      double target = p1 - PERCENT*(p1-best2);
      if(MathAbs(best3 - target) > (p1-best2)*0.05) return(false);
     }

   wave.active=true;
   wave.directionUp=up;
   wave.bar4=Bars(_Symbol,_Period)-1;
   wave.p1=p1; wave.p2=best2; wave.p3=best3; wave.p4=p4;
   wave.trade1=false; wave.trade2=false; wave.trade3=false;
   wave.p5=0; wave.p6=0;
   return(true);
  }

//+------------------------------------------------------------------+
//| Calculate lot size based on risk                                 |
//+------------------------------------------------------------------+
double CalcLot(double entry,double stop)
  {
   double riskMoney=AccountInfoDouble(ACCOUNT_EQUITY)*RiskPercent/100.0;
   double tickValue=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_VALUE);
   double tickSize =SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE);
   double stopPoints = MathAbs(entry-stop)/tickSize;
   if(stopPoints<=0.0 || tickValue<=0.0)
      return(0.0);
   double volume=riskMoney/(stopPoints*tickValue);
   double minLot=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN);
   double lotStep=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
   double maxLot=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MAX);
   volume= MathMax(minLot, MathMin(maxLot, NormalizeDouble(volume/lotStep,0)*lotStep));
   return(volume);
  }

//+------------------------------------------------------------------+
//| Add new trade info                                               |
//+------------------------------------------------------------------+
void AddTrade(ulong ticket,double sl)
  {
   TradeInfo info;
   info.ticket=ticket;
   info.originalSL=sl;
   info.openBar=Bars(_Symbol,_Period);
   info.stage=0;
   int n=ArraySize(g_trades);
   ArrayResize(g_trades,n+1);
   g_trades[n]=info;
  }

//+------------------------------------------------------------------+
//| Remove trade info by index                                       |
//+------------------------------------------------------------------+
void RemoveTrade(int index)
  {
   int total=ArraySize(g_trades);
   if(index<0 || index>=total) return;
   for(int i=index;i<total-1;i++)
      g_trades[i]=g_trades[i+1];
   ArrayResize(g_trades,total-1);
  }

//+------------------------------------------------------------------+
//| Manage open trades                                               |
//+------------------------------------------------------------------+
void ManageTrades()
  {
   int currentBars=Bars(_Symbol,_Period);
   for(int i=ArraySize(g_trades)-1;i>=0;i--)
     {
      TradeInfo info=g_trades[i];
      if(!PositionSelectByTicket(info.ticket))
        {
         RemoveTrade(i);
         continue;
        }
      int barsSince = currentBars - info.openBar;
      double entry = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl    = PositionGetDouble(POSITION_SL);
      double tp    = PositionGetDouble(POSITION_TP);
      ENUM_POSITION_TYPE type=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      double risk;
      if(type==POSITION_TYPE_BUY)
         risk = entry - info.originalSL;
      else
         risk = info.originalSL - entry;

      double newSL=sl;
      if(info.stage==0 && barsSince>=HALFSL)
        {
         if(type==POSITION_TYPE_BUY)
            newSL = entry - risk*0.5;
         else
            newSL = entry + risk*0.5;
         info.stage=1;
        }
      else if(info.stage==1 && barsSince>=2*HALFSL)
        {
         if(type==POSITION_TYPE_BUY)
            newSL = entry - risk*0.25;
         else
            newSL = entry + risk*0.25;
         info.stage=2;
        }
      else if(info.stage==2 && barsSince>=3*HALFSL)
        {
         newSL = entry;
         info.stage=3;
        }

      if(newSL!=sl)
         trade.PositionModify(info.ticket,newSL,tp);

      if(barsSince>=TCLOSE)
        {
         trade.PositionClose(info.ticket);
         RemoveTrade(i);
         continue;
        }

      g_trades[i]=info;
     }
  }

//+------------------------------------------------------------------+
//| Check trade setups after wave                                    |
//+------------------------------------------------------------------+
void CheckTrades(WaveInfo &wave)
  {
   int barsSinceWave = Bars(_Symbol,_Period)-1 - wave.bar4;
   if(barsSinceWave>TRADESCHECKING)
     {
      wave.active=false;
      return;
     }

   double price = GetAvgPrice(0);
   double atrBuf[];
   double atr=0.0;
   if(CopyBuffer(atr_handle,0,1,1,atrBuf)>0)
      atr=atrBuf[0];
   if(wave.p5==0.0)
     {
      // search for next peak or valley after wave.p4
      if(wave.directionUp && IsPeak(0))
        {
         wave.p5=price;
         wave.p6=wave.p3 + PERCENT*(wave.p5-wave.p3);
        }
      else if(!wave.directionUp && IsValley(0))
        {
         wave.p5=price;
         wave.p6=wave.p3 - PERCENT*(wave.p3-wave.p5);
        }
     }
   else
     {
      // trade 1
      if(!wave.trade1)
        {
         if( (wave.directionUp && price<=wave.p6) ||
             (!wave.directionUp && price>=wave.p6) )
           {
            double entry = wave.directionUp?SymbolInfoDouble(_Symbol,SYMBOL_ASK):SymbolInfoDouble(_Symbol,SYMBOL_BID);
            double sl = wave.directionUp ? wave.p3 - atr*STOPATR1 : wave.p3 + atr*STOPATR1;
            double lot = CalcLot(entry,sl);
            double risk = MathAbs(entry-sl);
            double tp = wave.directionUp ? entry + risk*RR1 : entry - risk*RR1;
            bool ok;
            if(wave.directionUp)
               ok=trade.Buy(lot,_Symbol,entry,sl,tp);
            else
               ok=trade.Sell(lot,_Symbol,entry,sl,tp);
            if(ok)
              {
               ulong ticket=trade.ResultOrder();
               AddTrade(ticket,sl);
              }
            wave.trade1=true;
           }
        }
      else if(!wave.trade2)
        {
         if( (wave.directionUp && price>wave.p6) ||
             (!wave.directionUp && price<wave.p6) )
           {
            double entry = wave.directionUp?SymbolInfoDouble(_Symbol,SYMBOL_ASK):SymbolInfoDouble(_Symbol,SYMBOL_BID);
            double sl = wave.directionUp ? wave.p3 - atr*STOPATR2 : wave.p3 + atr*STOPATR2;
            double lot = CalcLot(entry,sl);
            double risk = MathAbs(entry-sl);
            double tp = wave.directionUp ? entry + risk*RR2 : entry - risk*RR2;
            bool ok;
            if(wave.directionUp)
               ok=trade.Buy(lot,_Symbol,entry,sl,tp);
            else
               ok=trade.Sell(lot,_Symbol,entry,sl,tp);
            if(ok)
              {
               ulong ticket=trade.ResultOrder();
               AddTrade(ticket,sl);
              }
            wave.trade2=true;
           }
        }
      else if(!wave.trade3)
        {
         if( (wave.directionUp && price>=wave.p5) ||
             (!wave.directionUp && price<=wave.p5) )
           {
            double entry = wave.directionUp?SymbolInfoDouble(_Symbol,SYMBOL_ASK):SymbolInfoDouble(_Symbol,SYMBOL_BID);
            double sl = wave.directionUp ? wave.p3 - atr*STOPATR3 : wave.p3 + atr*STOPATR3;
            double lot = CalcLot(entry,sl);
            double risk = MathAbs(entry-sl);
            double tp = wave.directionUp ? entry + risk*RR3 : entry - risk*RR3;
            bool ok;
            if(wave.directionUp)
               ok=trade.Buy(lot,_Symbol,entry,sl,tp);
            else
               ok=trade.Sell(lot,_Symbol,entry,sl,tp);
            if(ok)
              {
               ulong ticket=trade.ResultOrder();
               AddTrade(ticket,sl);
              }
            wave.trade3=true;
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   static datetime lastBar=0;
   datetime current=iTime(_Symbol,_Period,0);
   if(current==lastBar)
     {
      ManageTrades();
      return;
     }
   lastBar=current;

   ManageTrades();

   // always check for new wave
   WaveInfo wtmp;
   if(DetectWave(wtmp))
      g_wave=wtmp;

   if(g_wave.active)
      CheckTrades(g_wave);
  }

//+------------------------------------------------------------------+
//| Calculate duration in bars and months                            |
//+------------------------------------------------------------------+
void CalcDuration(int &bars,double &months)
  {
   datetime end_time=TimeCurrent();
   bars = Bars(_Symbol,_Period,test_start,end_time);
   months = (double)(end_time - test_start)/(30.0*24*60*60);
   if(months<=0.0) months=1.0;
   if(bars<=0) bars=1;
  }

//+------------------------------------------------------------------+
//| Custom optimization criterion                                    |
//+------------------------------------------------------------------+
double OnTester()
  {
   int bars;
   double months;
   CalcDuration(bars,months);

   double totalTrades = TesterStatistics(STAT_TRADES);
   double totalProfit = TesterStatistics(STAT_PROFIT);
   double startEquity = TesterStatistics(STAT_INITIAL_DEPOSIT);
   double drawdown = TesterStatistics(STAT_EQUITY_DDREL_PERCENT)/100.0;

   double T = totalTrades / bars;
   double P = (totalProfit/startEquity)/months;
   double D = drawdown;

   double weightSum = Wt + Wp + Wd;
   double score = (T*Wt + P*Wp - D*Wd)/weightSum;

   return(score*100.0);
  }
//+------------------------------------------------------------------+
