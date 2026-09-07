package com.riversys.counter;

import android.os.Handler;
import android.os.Looper;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/** Polls the riverSys FastAPI /api/status endpoint. */
final class StatusPoller {
    interface Listener {
        void onSnapshot(DetectionSnapshot snapshot);

        void onError(String message);
    }

    private static final int INTERVAL_MS = 400;
    private static final int TIMEOUT_MS = 2500;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Handler main = new Handler(Looper.getMainLooper());
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final AtomicBoolean loopScheduled = new AtomicBoolean(false);
    private volatile String baseUrl = "";
    private Listener listener;

    void setListener(Listener listener) {
        this.listener = listener;
    }

    void start(String serverUrl) {
        baseUrl = normalize(serverUrl);
        running.set(true);
        if (loopScheduled.compareAndSet(false, true)) {
            executor.execute(() -> {
                try {
                    loop();
                } finally {
                    loopScheduled.set(false);
                }
            });
        }
    }

    void stop() {
        running.set(false);
    }

    void shutdown() {
        running.set(false);
        executor.shutdownNow();
    }

    static String normalize(String raw) {
        String url = raw == null ? "" : raw.trim();
        if (url.isEmpty()) {
            return "http://192.168.1.50:8000";
        }
        if (!url.startsWith("http://") && !url.startsWith("https://")) {
            url = "http://" + url;
        }
        if (url.endsWith("/")) {
            url = url.substring(0, url.length() - 1);
        }
        return url;
    }

    private void loop() {
        while (running.get()) {
            try {
                DetectionSnapshot snapshot = fetch();
                dispatchSnapshot(snapshot);
            } catch (Exception e) {
                dispatchError(e.getMessage() == null ? "Request failed" : e.getMessage());
            }
            try {
                Thread.sleep(INTERVAL_MS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }
    }

    private DetectionSnapshot fetch() throws Exception {
        URL url = new URL(baseUrl + "/api/status");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        try {
            conn.setRequestMethod("GET");
            conn.setConnectTimeout(TIMEOUT_MS);
            conn.setReadTimeout(TIMEOUT_MS);
            conn.setUseCaches(false);
            int code = conn.getResponseCode();
            InputStream stream = code >= 400 ? conn.getErrorStream() : conn.getInputStream();
            String body = readAll(stream);
            if (code != 200) {
                throw new IllegalStateException("HTTP " + code + " " + body);
            }
            return DetectionSnapshot.fromJson(new JSONObject(body));
        } finally {
            conn.disconnect();
        }
    }

    private static String readAll(InputStream stream) throws Exception {
        if (stream == null) {
            return "";
        }
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            StringBuilder out = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                out.append(line);
            }
            return out.toString();
        }
    }

    private void dispatchSnapshot(DetectionSnapshot snapshot) {
        Listener current = listener;
        if (current == null) {
            return;
        }
        main.post(() -> current.onSnapshot(snapshot));
    }

    private void dispatchError(String message) {
        Listener current = listener;
        if (current == null) {
            return;
        }
        main.post(() -> current.onError(message));
    }
}
