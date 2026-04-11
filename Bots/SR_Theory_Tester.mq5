//+------------------------------------------------------------------+
//|                                             SR_Theory_Tester.mq5 |
//|                                Analytics Bot to confirm crossing |
//+------------------------------------------------------------------+
#property strict

// --- Strategy Inputs ---
input int    LookbackCandles = 200;      // Histogram Lookback
input int    BinSizePoints = 50;         // Size of each bin in points
input int    MinBinTouches = 50;         // Minimum touches for SR Line
input int    AtrPeriod = 14;             // ATR Period
input double AtrMultiplier = 3.0;        // ATR Multiplier for buffer zone
input int    WaitBars = 500;             // Max bars to wait for a cross

// --- Global Tracking Variables ---
int totalTriggers = 0;
int crossedByClose = 0;
int atrHandle;
datetime lastBarTime;

// Structure to hold our active triggers waiting for 500 bars
struct TrackedTrigger {
    double srLine;
    int direction; // 1 = Buy (Price came from above), -1 = Sell
    int startBarIdx;
};
TrackedTrigger activeSetups[];

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
    atrHandle = iATR(_Symbol, _Period, AtrPeriod);
    if(atrHandle == INVALID_HANDLE) {
        Print("Failed to create ATR handle");
        return INIT_FAILED;
    }
    
    totalTriggers = 0;
    crossedByClose = 0;
    ArrayFree(activeSetups);
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    if(totalTriggers > 0) {
        double percentage = ((double)crossedByClose / totalTriggers) * 100.0;
        Print("==================================================");
        Print("TEST FINISHED: SR Theory Confirmation");
        Print("Total Valid Triggers Found: ", totalTriggers);
        Print("Triggers Crossed by Close within ", WaitBars, " bars: ", crossedByClose);
        Print("Crossing Probability: ", DoubleToString(percentage, 2), "%");
        Print("==================================================");
    } else {
        Print("No triggers found during this test period.");
    }
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
    datetime currentTime = iTime(_Symbol, _Period, 0);
    if(currentTime == lastBarTime) return; // Only run on new bars
    lastBarTime = currentTime;

    int currentBars = iBars(_Symbol, _Period);

    // 1. Check existing setups to see if they crossed
    for(int i = ArraySize(activeSetups) - 1; i >= 0; i--) {
        int elapsedBars = currentBars - activeSetups[i].startBarIdx;
        
        // If 500 bars passed without a cross, remove it
        if(elapsedBars > WaitBars) {
            ArrayRemove(activeSetups, i, 1);
            continue;
        }

        double close1 = iClose(_Symbol, _Period, 1); // Check completed bar
        bool crossed = false;
        
        // If Buy (price came from above), crossing means dropping below the SR line
        if(activeSetups[i].direction == 1 && close1 < activeSetups[i].srLine) crossed = true;
        
        // If Sell (price came from below), crossing means rising above the SR line
        if(activeSetups[i].direction == -1 && close1 > activeSetups[i].srLine) crossed = true;

        if(crossed) {
            crossedByClose++;
            ArrayRemove(activeSetups, i, 1); // Remove from tracking once confirmed
        }
    }

    // 2. Check for NEW setups
    CheckForTrigger(currentBars);
}

//+------------------------------------------------------------------+
//| Histogram and Trigger Logic                                      |
//+------------------------------------------------------------------+
void CheckForTrigger(int currentBarsCount) {
    // Need enough history
    if(currentBarsCount < LookbackCandles + 2) return;

    double atrArray[];
    if(CopyBuffer(atrHandle, 0, 1, 1, atrArray) <= 0) return;
    double bufferVal = atrArray[0] * AtrMultiplier;

    double high[], low[];
    CopyHigh(_Symbol, _Period, 2, LookbackCandles, high);
    CopyLow(_Symbol, _Period, 2, LookbackCandles, low);
    
    double open1 = iOpen(_Symbol, _Period, 1);
    double close1 = iClose(_Symbol, _Period, 1);

    double maxPrice = high[ArrayMaximum(high)];
    double minPrice = low[ArrayMinimum(low)];

    double binSize = BinSizePoints * _Point;
    if(binSize <= 0) return;
    
    int numBins = (int)MathCeil((maxPrice - minPrice) / binSize);
    if(numBins <= 0 || numBins > 10000) return; // safety catch

    int counts[];
    ArrayResize(counts, numBins);
    ArrayInitialize(counts, 0);

    // Populate Histogram Bins
    for(int i = 0; i < LookbackCandles; i++) {
        int startIdx = (int)MathFloor((low[i] - minPrice) / binSize);
        int endIdx = (int)MathFloor((high[i] - minPrice) / binSize);
        
        startIdx = MathMax(0, startIdx);
        endIdx = MathMin(numBins - 1, endIdx);
        
        for(int b = startIdx; b <= endIdx; b++) {
            counts[b]++;
        }
    }

    // Check SR lines for triggers
    for(int b = 0; b < numBins; b++) {
        if(counts[b] >= MinBinTouches) {
            double srLine = minPrice + (b * binSize) + (binSize / 2.0);

            bool openAbove = (open1 > srLine + bufferVal);
            bool openBelow = (open1 < srLine - bufferVal);
            bool closeInZone = (close1 <= srLine + bufferVal && close1 >= srLine - bufferVal);

            // If entering from clear air into the zone
            if(closeInZone) {
                int dir = 0;
                if(openAbove) dir = 1;       // Buy Setup
                else if(openBelow) dir = -1; // Sell Setup
                
                if(dir != 0) {
                    totalTriggers++;
                    int size = ArraySize(activeSetups);
                    ArrayResize(activeSetups, size + 1);
                    activeSetups[size].srLine = srLine;
                    activeSetups[size].direction = dir;
                    activeSetups[size].startBarIdx = currentBarsCount;
                    break; // Only trigger one setup per candle to prevent duplicates
                }
            }
        }
    }
}