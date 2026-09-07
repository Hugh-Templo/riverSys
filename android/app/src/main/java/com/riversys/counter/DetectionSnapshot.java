package com.riversys.counter;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

/** One poll of /api/status from the riverSys backend. */
final class DetectionSnapshot {
    final int liveCount;
    final int recyclableCount;
    final int nonRecyclableCount;
    final String detectorMode;
    final String decision;
    final List<Row> rows;

    static final class Row {
        final String label;
        final double confidence;
        final String source;
        final boolean recyclable;

        Row(String label, double confidence, String source, boolean recyclable) {
            this.label = label;
            this.confidence = confidence;
            this.source = source;
            this.recyclable = recyclable;
        }
    }

    DetectionSnapshot(
            int liveCount,
            int recyclableCount,
            int nonRecyclableCount,
            String detectorMode,
            String decision,
            List<Row> rows) {
        this.liveCount = liveCount;
        this.recyclableCount = recyclableCount;
        this.nonRecyclableCount = nonRecyclableCount;
        this.detectorMode = detectorMode;
        this.decision = decision;
        this.rows = rows;
    }

    static DetectionSnapshot fromJson(JSONObject data) {
        JSONArray detections = data.optJSONArray("detections");
        List<Row> rows = new ArrayList<>();
        int recyclable = 0;
        int nonRecyclable = 0;
        if (detections != null) {
            for (int i = 0; i < detections.length(); i++) {
                JSONObject item = detections.optJSONObject(i);
                if (item == null) {
                    continue;
                }
                String label = item.optString("label", "object");
                boolean isRecyclable = "recyclable".equalsIgnoreCase(label);
                if (isRecyclable) {
                    recyclable++;
                } else {
                    nonRecyclable++;
                }
                rows.add(new Row(
                        label,
                        item.optDouble("confidence", 0),
                        item.optString("source", ""),
                        isRecyclable));
            }
        }
        String decision = data.optString("decision", "");
        if ("null".equals(decision)) {
            decision = "";
        }
        return new DetectionSnapshot(
                rows.size(),
                recyclable,
                nonRecyclable,
                data.optString("detector_mode", "—"),
                decision,
                rows);
    }
}
