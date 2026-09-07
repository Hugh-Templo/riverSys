package com.riversys.counter;

import android.content.SharedPreferences;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;

import com.riversys.counter.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity implements StatusPoller.Listener {
    private static final String PREFS = "object_counter";
    private static final String KEY_URL = "server_url";
    private static final String DEFAULT_URL = "http://192.168.1.50:8000";

    private ActivityMainBinding binding;
    private final StatusPoller poller = new StatusPoller();

    private boolean connected;
    private int previousLiveCount;
    private int sessionTotal;
    private int sessionRecyclable;
    private int sessionNonRecyclable;
    private DetectionSnapshot latest;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        WindowCompat.setDecorFitsSystemWindows(getWindow(), false);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        applyInsets();

        SharedPreferences prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        binding.serverUrl.setText(prefs.getString(KEY_URL, DEFAULT_URL));

        poller.setListener(this);
        binding.connectButton.setOnClickListener(v -> toggleConnection());
        binding.resetButton.setOnClickListener(v -> resetTotals());
        binding.addCurrentButton.setOnClickListener(v -> addCurrentToTotal());
        renderCounts();
        showEmptyList();
    }

    @Override
    protected void onDestroy() {
        poller.shutdown();
        super.onDestroy();
    }

    private void applyInsets() {
        ViewCompat.setOnApplyWindowInsetsListener(binding.getRoot(), (view, insets) -> {
            Insets bars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            view.setPadding(view.getPaddingLeft(), bars.top + 8, view.getPaddingRight(), bars.bottom + 8);
            return insets;
        });
    }

    private void toggleConnection() {
        if (connected) {
            disconnect();
            return;
        }
        String url = StatusPoller.normalize(String.valueOf(binding.serverUrl.getText()));
        getSharedPreferences(PREFS, MODE_PRIVATE).edit().putString(KEY_URL, url).apply();
        hideKeyboard();
        connected = true;
        previousLiveCount = 0;
        latest = null;
        binding.connectButton.setText(R.string.disconnect);
        binding.statusLine.setText(R.string.status_connecting);
        poller.start(url);
    }

    private void disconnect() {
        connected = false;
        poller.stop();
        binding.connectButton.setText(R.string.connect);
        binding.statusLine.setText(R.string.status_idle);
        binding.modeLine.setText("detector: —");
        binding.liveCount.setText("0");
        latest = null;
        previousLiveCount = 0;
        showEmptyList();
    }

    private void resetTotals() {
        sessionTotal = 0;
        sessionRecyclable = 0;
        sessionNonRecyclable = 0;
        previousLiveCount = latest == null ? 0 : latest.liveCount;
        renderCounts();
        Toast.makeText(this, "Totals reset", Toast.LENGTH_SHORT).show();
    }

    private void addCurrentToTotal() {
        if (latest == null || latest.liveCount == 0) {
            Toast.makeText(this, "Nothing to add", Toast.LENGTH_SHORT).show();
            return;
        }
        addSnapshot(latest);
        previousLiveCount = latest.liveCount;
        renderCounts();
    }

    @Override
    public void onSnapshot(DetectionSnapshot snapshot) {
        if (!connected) {
            return;
        }
        latest = snapshot;
        if (binding.autoCountSwitch.isChecked()
                && previousLiveCount == 0
                && snapshot.liveCount > 0) {
            addSnapshot(snapshot);
        }
        previousLiveCount = snapshot.liveCount;
        binding.liveCount.setText(String.valueOf(snapshot.liveCount));
        binding.liveCaption.setText(snapshot.liveCount == 1 ? "object" : "objects");
        renderCounts();
        renderList(snapshot);
        String decision = snapshot.decision.isEmpty() ? "waiting" : snapshot.decision;
        binding.statusLine.setText("Connected · " + decision.replace('_', ' '));
        binding.modeLine.setText("detector: " + snapshot.detectorMode);
    }

    @Override
    public void onError(String message) {
        if (!connected) {
            return;
        }
        binding.statusLine.setText("Cannot reach Pi: " + message);
    }

    private void addSnapshot(DetectionSnapshot snapshot) {
        sessionTotal += snapshot.liveCount;
        sessionRecyclable += snapshot.recyclableCount;
        sessionNonRecyclable += snapshot.nonRecyclableCount;
    }

    private void renderCounts() {
        binding.sessionTotal.setText(String.valueOf(sessionTotal));
        binding.recyclableTotal.setText(String.valueOf(sessionRecyclable));
        binding.nonRecyclableTotal.setText(String.valueOf(sessionNonRecyclable));
    }

    private void renderList(DetectionSnapshot snapshot) {
        LinearLayout list = binding.detectionList;
        list.removeAllViews();
        if (snapshot.rows.isEmpty()) {
            showEmptyList();
            return;
        }
        LayoutInflater inflater = getLayoutInflater();
        for (DetectionSnapshot.Row row : snapshot.rows) {
            View item = inflater.inflate(R.layout.item_detection, list, false);
            TextView label = item.findViewById(R.id.label);
            TextView confidence = item.findViewById(R.id.confidence);
            View swatch = item.findViewById(R.id.swatch);
            String pretty = row.label.replace('_', ' ');
            if (!row.source.isEmpty()) {
                pretty += "  ·  " + row.source;
            }
            label.setText(pretty);
            confidence.setText(Math.round(row.confidence * 100) + "%");
            int color = ContextCompat.getColor(
                    this, row.recyclable ? R.color.recyclable : R.color.non_recyclable);
            GradientDrawable dot = new GradientDrawable();
            dot.setShape(GradientDrawable.OVAL);
            dot.setColor(color);
            swatch.setBackground(dot);
            list.addView(item);
        }
    }

    private void showEmptyList() {
        binding.detectionList.removeAllViews();
        TextView empty = new TextView(this);
        empty.setText(R.string.no_objects);
        empty.setTextColor(ContextCompat.getColor(this, R.color.muted));
        empty.setTextSize(14);
        binding.detectionList.addView(empty);
    }

    private void hideKeyboard() {
        InputMethodManager imm = (InputMethodManager) getSystemService(INPUT_METHOD_SERVICE);
        if (imm != null && getCurrentFocus() != null) {
            imm.hideSoftInputFromWindow(getCurrentFocus().getWindowToken(), 0);
        }
    }
}
