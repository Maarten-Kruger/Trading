#property copyright "MJ Kruger"
#property link      "https://github.com/Maarten-Kruger/Trading"
#property version   "1.0"
#property strict

#include <Trade/Trade.mqh>
#include <Trade/PositionInfo.mqh>

//---- input parameters
input int      INTERVAL   = 60;     // seconds between price samples
input int      BACKTIME   = 20;     // number of samples to look back for wave
input double   PERCENT    = 0.618;  // retracement percent
input int      WAVECHECK  = 50;     // max intervals since last wave to allow trades
input double   STOPATR1   = 1.0;    // ATR multiple for Wise Man 1 stop
input double   STOPATR2   = 1.5;    // ATR multiple for Wise Man 2 stop
input double   STOPATR3   = 2.0;    // ATR multiple for Wise Man 3 stop
input double   RR1        = 1.0;    // risk reward for Wise Man 1
input double   RR2        = 1.5;    // risk reward for Wise Man 2
input double   RR3        = 2.0;    // risk reward for Wise Man 3
input int      HALF       = 10;     // intervals between stop reductions
input int      TCLOSE     = 100;    // close trade after this many intervals
input double   VOL        = 0.1;    // lot size
input int      ATRPERIOD  = 14;     // ATR period
input long     MAGIC      = 20050917;

//---- trading objects
CTrade trade;
CPositionInfo posinfo;
int atr_handle = INVALID_HANDLE;

double prices[];          // recorded prices
int    last_wave_index = -1000; // index when last wave was detected

// wave points
bool   wave_up = true;
double P1, P2, P3, P4, P5, P6;
int    idx_P4 = -1;

// trade tracking
struct TradeInfo
  {
   ulong    ticket;
   datetime open_time;
   double   entry;
   double   init_sl;
   int      stage;     // 0:not moved,1:50%,2:25%,3:breakeven
  };

TradeInfo wm1={0}, wm2={0}, wm3={0};

//---- helper to get ATR value
bool GetATR(double &atr)
  {
   if(atr_handle==INVALID_HANDLE) return(false);
   double buf[];
   if(CopyBuffer(atr_handle,0,0,1,buf)<=0) return(false);
   atr = buf[0];
   return(true);
  }

//---- detect wave pattern on last BACKTIME prices
bool DetectWave()
  {
   int total = ArraySize(prices);
   if(total < BACKTIME)
      return(false);
   int start = total - BACKTIME;
   P1 = prices[start];
   P4 = prices[total-1];

   // find P2 and P3
   int p2_index = start+1;
   double p2_val = prices[p2_index];
   for(int i=start+1;i<total-1;i++)
     {
      if(prices[i]>p2_val){p2_val=prices[i]; p2_index=i;}
     }
   int p3_index = p2_index+1;
   double p3_val = prices[p3_index];
   for(int i=p2_index+1;i<total-1;i++)
     {
      if(prices[i]<p3_val){p3_val=prices[i]; p3_index=i;}
     }

   // up wave check
   bool up = true;
   for(int i=start;i<total;i++)
     {
      if(i!=start && prices[i]<=P1) up=false;
      if(i!=total-1 && prices[i]>=P4) up=false;
     }
   double expect_p3 = P1 + PERCENT*(p2_val-P1);
   double tol = _Point*10;
   if(up && p3_val<p2_val && MathAbs(p3_val-expect_p3)<=tol)
     {
      wave_up = true;
      P2 = p2_val; P3 = p3_val; idx_P4 = total-1; P5 = P4; P6 = P3 + PERCENT*(P5-P3);
      last_wave_index = total-1;
      return(true);
     }

   // down wave check
   up = true;
   for(int i=start;i<total;i++)
     {
      if(i!=start && prices[i]>=P1) up=false;
      if(i!=total-1 && prices[i]<=P4) up=false;
     }
   expect_p3 = P1 + PERCENT*(p2_val-P1); // for down wave p2_val is low later
   if(!up)
     {
      // recompute for down wave
      p2_val = prices[start+1]; p2_index=start+1;
      for(int i=start+1;i<total-1;i++)
        if(prices[i]<p2_val){p2_val=prices[i]; p2_index=i;}
      p3_val = prices[p2_index+1]; p3_index=p2_index+1;
      for(int i=p2_index+1;i<total-1;i++)
        if(prices[i]>p3_val){p3_val=prices[i]; p3_index=i;}
      bool down=true;
      for(int i=start;i<total;i++)
        {
         if(i!=start && prices[i]>=P1) down=false;
         if(i!=total-1 && prices[i]<=P4) down=false;
        }
      expect_p3 = P1 + PERCENT*(p2_val-P1);
      if(down && p3_val>p2_val && MathAbs(p3_val-expect_p3)<=tol)
        {
         wave_up = false;
         P2=p2_val; P3=p3_val; idx_P4=total-1; P5=P4; P6=P3 + PERCENT*(P5-P3);
         last_wave_index = total-1;
         return(true);
        }
     }
   return(false);
  }

//---- open trade helper
bool OpenTrade(TradeInfo &t,double sl,double tp)
  {
   trade.SetDeviationInPoints(20);
   trade.SetExpertMagicNumber(MAGIC);
   bool res=false;
   if(wave_up) res=trade.Buy(VOL,_Symbol,0,sl,tp,"WiseMan");
   else        res=trade.Sell(VOL,_Symbol,0,sl,tp,"WiseMan");
   if(res)
     {
      t.ticket=trade.ResultDeal();
      t.open_time=TimeCurrent();
      t.entry=(wave_up?trade.ResultPrice():trade.ResultPrice());
      t.init_sl=sl;
      t.stage=0;
     }
   return(res);
  }

//---- update stop loss stages
void ManageTrade(TradeInfo &t)
  {
   if(t.ticket==0) return;
   if(!posinfo.Select((long)t.ticket)) return;
   double entry = posinfo.PriceOpen();
   double sl    = posinfo.StopLoss();
   double atr=0; if(!GetATR(atr)) return;
   int elapsed = (int)((TimeCurrent()-t.open_time)/INTERVAL);
   double risk = MathAbs(entry - t.init_sl);
   if(elapsed>=HALF && t.stage==0)
     { // reduce risk to 50%
      double new_sl = (wave_up? entry - risk/2.0 : entry + risk/2.0);
      trade.PositionModify(posinfo.Symbol(), new_sl, posinfo.TakeProfit());
      t.stage=1;
     }
   if(elapsed>=HALF*2 && t.stage==1)
     { // reduce risk to 25%
      double new_sl = (wave_up? entry - risk/4.0 : entry + risk/4.0);
      trade.PositionModify(posinfo.Symbol(), new_sl, posinfo.TakeProfit());
      t.stage=2;
     }
   if(elapsed>=HALF*3 && t.stage==2)
     {
      double new_sl = entry; // breakeven
      trade.PositionModify(posinfo.Symbol(), new_sl, posinfo.TakeProfit());
      t.stage=3;
     }
   if(elapsed>=TCLOSE)
     {
      trade.PositionClose((long)t.ticket);
      t.ticket=0;
     }
  }

//---- OnInit
int OnInit()
  {
   trade.SetExpertMagicNumber(MAGIC);
   atr_handle = iATR(_Symbol,_Period,ATRPERIOD);
   EventSetTimer(INTERVAL);
   return(INIT_SUCCEEDED);
  }

//---- OnDeinit
void OnDeinit(const int reason)
  {
   EventKillTimer();
  }

//---- OnTimer: main logic
void OnTimer()
  {
   double price=0.0;
   if(!SymbolInfoDouble(_Symbol, SYMBOL_BID, price)) return;
   ArrayPush(prices,price);

   // detect wave
   DetectWave();

   int total = ArraySize(prices);
   if(idx_P4>=0)
     {
      int since_wave = total - last_wave_index;
      if(since_wave <= WAVECHECK)
        {
         // update P5 and P6
         double last_price = prices[total-1];
         if(wave_up)
           {
            if(last_price>P5) {P5=last_price; P6=P3 + PERCENT*(P5-P3);}      
            // Wise Man 1
            if(wm1.ticket==0 && last_price>=P6)
              {
               double atr=0; if(!GetATR(atr)) return; 
               double sl = P3 - STOPATR1*atr;
               double risk = last_price - sl;
               double tp = last_price + RR1*risk;
               if(OpenTrade(wm1,sl,tp)){}
              }
            // Wise Man 2
            if(wm1.ticket!=0 && wm2.ticket==0 && last_price>wm1.entry)
              {
               double atr=0; if(!GetATR(atr)) return;
               double sl = P3 - STOPATR2*atr;
               double risk = last_price - sl;
               double tp = last_price + RR2*risk;
               if(OpenTrade(wm2,sl,tp)){}
              }
            // Wise Man 3
            if(wm3.ticket==0 && last_price>P5)
              {
               double atr=0; if(!GetATR(atr)) return;
               double sl = P3 - STOPATR3*atr;
               double risk = last_price - sl;
               double tp = last_price + RR3*risk;
               if(OpenTrade(wm3,sl,tp)){}
              }
           }
         else
           {
            if(last_price<P5) {P5=last_price; P6=P3 + PERCENT*(P5-P3);}      
            if(wm1.ticket==0 && last_price<=P6)
              {
               double atr=0; if(!GetATR(atr)) return; 
               double sl = P3 + STOPATR1*atr;
               double risk = sl - last_price;
               double tp = last_price - RR1*risk;
               if(OpenTrade(wm1,sl,tp)){}
              }
            if(wm1.ticket!=0 && wm2.ticket==0 && last_price<wm1.entry)
              {
               double atr=0; if(!GetATR(atr)) return; 
               double sl = P3 + STOPATR2*atr;
               double risk = sl - last_price;
               double tp = last_price - RR2*risk;
               if(OpenTrade(wm2,sl,tp)){}
              }
            if(wm3.ticket==0 && last_price<P5)
              {
               double atr=0; if(!GetATR(atr)) return; 
               double sl = P3 + STOPATR3*atr;
               double risk = sl - last_price;
               double tp = last_price - RR3*risk;
               if(OpenTrade(wm3,sl,tp)){}
              }
           }
        }
     }

   // manage trades
   ManageTrade(wm1);
   ManageTrade(wm2);
   ManageTrade(wm3);
  }

