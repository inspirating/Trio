import Foundation

public struct PredictionGlucoseValue {
    public let date: Date
    public let value: Double

    public init(date: Date, value: Double) {
        self.date = date
        self.value = value
    }
}

public final class FullyAuditedBGPredictor {
    
    public struct PredictionResult {
        public let predictedValue: Double
        public let isLowAlertTriggered: Bool
        public let isHighAlertTriggered: Bool
        public let confidenceScore: Double
        public let minutesToLow: Double?
        public let minutesToHigh: Double?
    }
    
    // MARK: - Trend Line Models (ported from xDrip Forecast.java)
    
    private protocol TrendLine {
        mutating func setValues(y: [Double], x: [Double])
        func predict(x: Double) -> Double
        var errorVariance: Double? { get }
    }
    
    // Base OLS implementation for 2-parameter models (β₀ + β₁·x₁)
    // Ported from: Forecast.java → OLSTrendLine
    // Uses mean-centering for numerical stability with millisecond timestamps
    private class OLSTrendLine: TrendLine {
        private var coef0: Double = 0
        private var coef1: Double = 0
        private var lastErrorRate: Double? = nil
        
        func xVector(_ x: Double) -> Double {
            return x
        }
        
        func logY() -> Bool {
            return false
        }
        
        func setValues(y: [Double], x: [Double]) {
            let n = x.count
            guard n >= 2 else { return }
            
            var fitY = y
            if logY() {
                fitY = y.map { log($0) }
            }
            
            var tx = [Double]()
            for xi in x {
                tx.append(xVector(xi))
            }
            
            // Mean-centering for numerical stability with large x values (millisecond timestamps)
            let xMean = tx.reduce(0, +) / Double(n)
            let yMean = fitY.reduce(0, +) / Double(n)
            
            var sumXYc = 0.0, sumXXc = 0.0
            for i in 0..<n {
                let dx = tx[i] - xMean
                let dy = fitY[i] - yMean
                sumXYc += dx * dy
                sumXXc += dx * dx
            }
            
            guard abs(sumXXc) > 1e-30 else { return }
            
            coef1 = sumXYc / sumXXc
            coef0 = yMean - coef1 * xMean
            
            var sse = 0.0
            for i in 0..<n {
                let yHat = coef0 + coef1 * tx[i]
                let residual = fitY[i] - yHat
                sse += residual * residual
            }
            
            let df = Double(n) - 2
            lastErrorRate = df > 0 ? sse / df : 0
        }
        
        func predict(x: Double) -> Double {
            let x1 = xVector(x)
            let yhat = coef0 + coef1 * x1
            if logY() {
                return exp(yhat)
            }
            return yhat
        }
        
        var errorVariance: Double? {
            return lastErrorRate
        }
    }
    
    // PolyTrendLine(degree=1): y = β₀ + β₁·x  (Linear)
    private class PolyTrendLine1: OLSTrendLine {
        override func xVector(_ x: Double) -> Double {
            return x
        }
        
        override func logY() -> Bool {
            return false
        }
    }
    
    // LogTrendLine: y = β₀ + β₁·ln(x)
    private class LogTrendLine: OLSTrendLine {
        override func xVector(_ x: Double) -> Double {
            return log(x)
        }
        
        override func logY() -> Bool {
            return false
        }
    }
    
    // ExpTrendLine: ln(y) = β₀ + β₁·x  →  y = e^(β₀+β₁·x)
    private class ExpTrendLine: OLSTrendLine {
        override func xVector(_ x: Double) -> Double {
            return x
        }
        
        override func logY() -> Bool {
            return true
        }
    }
    
    // PowerTrendLine: ln(y) = β₀ + β₁·ln(x)  →  y = e^(β₀)·x^β₁
    private class PowerTrendLine: OLSTrendLine {
        override func xVector(_ x: Double) -> Double {
            return log(x)
        }
        
        override func logY() -> Bool {
            return true
        }
    }
    
    // MARK: - Main Prediction Engine
    // Ported from: BgGraphBuilder.java → addBgReadingValues() + low_occurs_at calculation
    
    public static func evaluate(
        history: [PredictionGlucoseValue],
        forecastMinutes: Double,
        lowThreshold: Double,
        highThreshold: Double
    ) -> PredictionResult? {
        
        let sortedHistory = history.sorted { $0.date < $1.date }
        guard sortedHistory.count >= 3 else { return nil }
        
        let currentDate = sortedHistory.last!.date
        let currentBG = sortedHistory.last!.value
        
        // Data window: 12 minutes base with dynamic expansion
        // Ported from: BgGraphBuilder.java line 1219-1227
        // trend_start_working = now - (1000 * 60 * 12)
        // if (ms_since_last_reading < 500000) trend_start_working -= ms_since_last_reading
        let now = Date()
        let msSinceLastReading = max(0, now.timeIntervalSince(currentDate) * 1000.0)
        var dataWindowStart = currentDate.addingTimeInterval(-12 * 60)
        if msSinceLastReading < 500_000 {
            dataWindowStart = dataWindowStart.addingTimeInterval(-msSinceLastReading / 1000.0)
        }
        
        let windowData = sortedHistory.filter { $0.date >= dataWindowStart }
        guard windowData.count >= 3 else { return nil }
        
        // Use millisecond timestamps for x values (matching xDrip's bgReading.timestamp)
        var xArray = [Double]()
        var yArray = [Double]()
        
        for point in windowData {
            let xVal = point.date.timeIntervalSince1970 * 1000.0
            xArray.append(xVal)
            yArray.append(point.value)
        }
        
        // Fit all 4 models and select best by error variance (matching xDrip)
        var models: [TrendLine] = [
            PolyTrendLine1(),
            LogTrendLine(),
            ExpTrendLine(),
            PowerTrendLine()
        ]
        
        var bestModel: TrendLine? = nil
        var minError: Double = 1e30
        
        for i in 0..<models.count {
            models[i].setValues(y: yArray, x: xArray)
            if let errVar = models[i].errorVariance, errVar >= 0 && errVar < minError {
                minError = errVar
                bestModel = models[i]
            }
        }
        
        guard let poly = bestModel else { return nil }
        
        let totalCount = sortedHistory.count
        let totalDuration = currentDate.timeIntervalSince(sortedHistory.first!.date) / 60.0
        let confidence = min(Double(totalCount) / ((totalDuration / 5.0) + 1), 1.0)
        
        // Predict value at forecastMinutes from now
        let nowMs = now.timeIntervalSince1970 * 1000.0
        let forecastTimeMs = nowMs + (forecastMinutes * 60 * 1000.0)
        let predictedValue = poly.predict(x: forecastTimeMs)
        
        // MARK: - Low Predictor (ported from BgGraphBuilder.java low_occurs_at logic)
        // Scan from max lookahead (99 minutes) backwards in 2.5-min steps
        // Uses plow_now = JoH.ts() (current system time), not last reading time
        
        var finalMinutesToLow: Double? = nil
        var finalMinutesToHigh: Double? = nil
        
        // Ported from: BgGraphBuilder.java line 1576-1600
        let plowNowMs = nowMs
        let lookaheadMs = plowNowMs + 99 * 60 * 1000.0
        let stepMs: Double = 1000 * 30 * 5 // 150000ms = 2.5 min (matching xDrip)
        
        // Low threshold crossing
        let predictedAtLookahead = poly.predict(x: lookaheadMs)
        
        if predictedAtLookahead <= lowThreshold {
            var lowOccursAtMs = lookaheadMs
            var scanMs = lookaheadMs
            while scanMs > plowNowMs {
                scanMs -= stepMs
                let predVal = poly.predict(x: scanMs)
                if predVal <= lowThreshold {
                    lowOccursAtMs = scanMs
                }
            }
            
            let minutesToLow = (lowOccursAtMs - plowNowMs) / (60 * 1000.0)
            if minutesToLow > 0 {
                finalMinutesToLow = minutesToLow
            }
        }
        
        // High threshold crossing
        if predictedAtLookahead >= highThreshold {
            var highOccursAtMs = lookaheadMs
            var scanMs = lookaheadMs
            while scanMs > plowNowMs {
                scanMs -= stepMs
                let predVal = poly.predict(x: scanMs)
                if predVal >= highThreshold {
                    highOccursAtMs = scanMs
                }
            }
            
            let minutesToHigh = (highOccursAtMs - plowNowMs) / (60 * 1000.0)
            if minutesToHigh > 0 {
                finalMinutesToHigh = minutesToHigh
            }
        }
        
        return PredictionResult(
            predictedValue: max(min(predictedValue, 33.3), 1.1),
            isLowAlertTriggered: predictedValue <= lowThreshold,
            isHighAlertTriggered: predictedValue >= highThreshold,
            confidenceScore: confidence,
            minutesToLow: finalMinutesToLow,
            minutesToHigh: finalMinutesToHigh
        )
    }
}
