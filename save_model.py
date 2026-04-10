import joblib
import model  # ملف التدريب عندك

# =========================
# Bundle Clean
# =========================
final_bundle = {
    # Models
    "dt_model": model.dt_model,
    "xgb_model": model.xgb_model,
    "voting_model": model.voting_model,
    "stacking_model": model.stacking_model,

    # Thresholds
    "thresholds": {
        "dt": model.dt_t,
        "xgb": model.xgb_t,
        "voting": model.voting_t,
        "stacking": model.stack_t
    }
}

# =========================
# Save
# =========================
joblib.dump(final_bundle, "final_models.pkl")

print("Models saved successfully ✅")