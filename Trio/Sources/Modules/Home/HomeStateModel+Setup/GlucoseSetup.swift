import CoreData
import Foundation
import SwiftUI
import UserNotifications

extension Home.StateModel {
    func setupGlucoseArray() {
        Task {
            do {
                let ids = try await self.fetchGlucose()
                let glucoseObjects: [GlucoseStored] = try await CoreDataStack.shared
                    .getNSManagedObject(with: ids, context: viewContext)
                await updateGlucoseArray(with: glucoseObjects)
            } catch {
                debug(
                    .default,
                    "\(DebuggingIdentifiers.failed) Error setting up glucose array: \(error)"
                )
            }
        }
    }

    private func fetchGlucose() async throws -> [NSManagedObjectID] {
        let results = try await CoreDataStack.shared.fetchEntitiesAsync(
            ofType: GlucoseStored.self,
            onContext: glucoseFetchContext,
            predicate: NSPredicate.glucose,
            key: "date",
            ascending: true,
            batchSize: 50
        )

        return try await glucoseFetchContext.perform {
            guard let fetchedResults = results as? [GlucoseStored] else {
                throw CoreDataError.fetchError(function: #function, file: #file)
            }

            // Update Main Chart Y Axis Values
            // Perform everything on "context" to be thread safe
            self.yAxisChartData(glucoseValues: fetchedResults)

            return fetchedResults.map(\.objectID)
        }
    }

    @MainActor private func updateGlucoseArray(with objects: [GlucoseStored]) {
        glucoseFromPersistence = objects
        latestTwoGlucoseValues = Array(objects.suffix(2))
        updatePredictionWarning(glucose: objects)
    }

    private func updatePredictionWarning(glucose: [GlucoseStored]) {
        let historyWindow = Date().addingTimeInterval(-45 * 60)
        let recentGlucose = glucose
            .filter { ($0.date ?? .distantPast) > historyWindow }
            .compactMap { entry -> PredictionGlucoseValue? in
                guard let date = entry.date else { return nil }
                let mmolValue = Double(entry.glucose) * Double(GlucoseUnits.exchangeRate)
                return PredictionGlucoseValue(date: date, value: mmolValue)
            }

        guard recentGlucose.count >= 4 else {
            debug(.default, "[预测调试] 血糖数据不足4个: \(recentGlucose.count)")
            predictionWarning = nil
            return
        }

        let lowThreshold = Double(truncating: settingsManager.settings.low as NSNumber) * Double(GlucoseUnits.exchangeRate)
        let highThreshold = Double(truncating: settingsManager.settings.high as NSNumber) * Double(GlucoseUnits.exchangeRate)

        debug(.default, """
            [预测调试] 数据点: \(recentGlucose.count)个, \
            低阈值: \(String(format: "%.2f", lowThreshold)), \
            高阈值: \(String(format: "%.2f", highThreshold))
            """)
        for (i, point) in recentGlucose.enumerated() {
            debug(.default, "[预测调试]   [\(i)] \(point.date) -> \(String(format: "%.2f", point.value))")
        }

        guard let result = FullyAuditedBGPredictor.evaluate(
            history: recentGlucose,
            forecastMinutes: 120,
            lowThreshold: lowThreshold,
            highThreshold: highThreshold
        ) else {
            debug(.default, "[预测调试] evaluate返回nil")
            predictionWarning = nil
            return
        }

        debug(.default, """
            [预测调试] evaluate完成: \
            predictedValue=\(String(format: "%.4f", result.predictedValue)), \
            minutesToLow=\(result.minutesToLow.map { String(format: "%.1f", $0) } ?? "nil"), \
            minutesToHigh=\(result.minutesToHigh.map { String(format: "%.1f", $0) } ?? "nil"), \
            isLowAlertTriggered=\(result.isLowAlertTriggered), \
            isHighAlertTriggered=\(result.isHighAlertTriggered), \
            confidence=\(String(format: "%.3f", result.confidenceScore))
            """)

        if let minutesToLow = result.minutesToLow, minutesToLow > 0 {
            let minutes = Int(minutesToLow.rounded(.up))
            predictionWarning = String(format: "还有%d分钟低血糖", minutes)
            warningColor = .red
            if minutesToLow <= 30 {
                sendLowGlucoseNotification(minutes: minutes)
            }
        } else if let minutesToHigh = result.minutesToHigh, minutesToHigh > 0 {
            let minutes = Int(minutesToHigh.rounded(.up))
            predictionWarning = String(format: "还有%d分钟高血糖", minutes)
            warningColor = .yellow
        } else {
            predictionWarning = String(format: "预测2H后血糖为%.2f mmol/L，正常", result.predictedValue)
            warningColor = .white
        }
    }

    private func sendLowGlucoseNotification(minutes: Int) {
        let center = UNUserNotificationCenter.current()
        center.getNotificationSettings { settings in
            guard settings.authorizationStatus == .authorized || settings.authorizationStatus == .provisional else { return }

            let content = UNMutableNotificationContent()
            content.title = "低血糖预警"
            content.body = String(format: "还有%d分钟低血糖", minutes)
            content.sound = .defaultCritical
            content.categoryIdentifier = NotificationCategoryIdentifier.trioAlert.rawValue

            let request = UNNotificationRequest(
                identifier: "trio.prediction.lowGlucose",
                content: content,
                trigger: nil
            )

            center.add(request) { error in
                if let error = error {
                    debug(.default, "[预测通知] 发送失败: \(error.localizedDescription)")
                }
            }
        }
    }
}
