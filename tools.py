"""
Tool implementations for the tool-calling evaluation.

Each tool is a plain Python function (the actual implementation) plus a shared JSON schema
dict that is converted to either OpenAI tool format or Gemini FunctionDeclaration as needed.
"""

from typing import Any, Dict


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def calculate_qsofa(respiratory_rate: int, altered_mentation: int, systolic_bp: int) -> Dict:
    """Calculate qSOFA (quick Sepsis-related Organ Failure Assessment) score."""
    score = 0
    components = []

    if respiratory_rate >= 22:
        score += 1
        components.append(f"RR {respiratory_rate}/min ≥22 → +1")
    else:
        components.append(f"RR {respiratory_rate}/min <22 → 0")

    if altered_mentation:
        score += 1
        components.append("Altered mentation (GCS <15) → +1")
    else:
        components.append("Normal mentation → 0")

    if systolic_bp <= 100:
        score += 1
        components.append(f"SBP {systolic_bp} mmHg ≤100 → +1")
    else:
        components.append(f"SBP {systolic_bp} mmHg >100 → 0")

    if score >= 2:
        interpretation = "HIGH RISK – likely sepsis; urgent ICU assessment, cultures, antibiotics, lactate"
    elif score == 1:
        interpretation = "INTERMEDIATE – monitor closely for deterioration"
    else:
        interpretation = "LOW RISK – sepsis less likely at this time"

    return {
        "qsofa_score": score,
        "max_score": 3,
        "components": components,
        "interpretation": interpretation,
    }


def calculate_sofa(
    pf_ratio: float,
    platelets: int,
    bilirubin: float,
    map_mmhg: float,
    vasopressor_dose: float,
    vasopressor_drug: str,
    gcs: int,
    creatinine: float,
    urine_output_ml_per_day: float,
) -> Dict:
    """
    Calculate SOFA (Sequential Organ Failure Assessment) score.

    pf_ratio: PaO2/FiO2 ratio
    platelets: platelet count x10^3/µL
    bilirubin: mg/dL
    map_mmhg: mean arterial pressure mmHg (set 0 if on vasopressors)
    vasopressor_dose: dose in mcg/kg/min (0 if none)
    vasopressor_drug: 'none', 'dopamine', 'dobutamine', 'norepinephrine', 'epinephrine'
    gcs: Glasgow Coma Scale 3-15
    creatinine: mg/dL
    urine_output_ml_per_day: mL/day
    """
    scores = {}

    # Respiratory
    if pf_ratio >= 400:
        scores["respiratory"] = 0
    elif pf_ratio >= 300:
        scores["respiratory"] = 1
    elif pf_ratio >= 200:
        scores["respiratory"] = 2
    elif pf_ratio >= 100:
        scores["respiratory"] = 3
    else:
        scores["respiratory"] = 4

    # Coagulation
    if platelets >= 150:
        scores["coagulation"] = 0
    elif platelets >= 100:
        scores["coagulation"] = 1
    elif platelets >= 50:
        scores["coagulation"] = 2
    elif platelets >= 20:
        scores["coagulation"] = 3
    else:
        scores["coagulation"] = 4

    # Liver
    if bilirubin < 1.2:
        scores["liver"] = 0
    elif bilirubin <= 1.9:
        scores["liver"] = 1
    elif bilirubin <= 5.9:
        scores["liver"] = 2
    elif bilirubin <= 11.9:
        scores["liver"] = 3
    else:
        scores["liver"] = 4

    # Cardiovascular
    drug = vasopressor_drug.lower()
    if vasopressor_dose == 0 or drug == "none":
        scores["cardiovascular"] = 0 if map_mmhg >= 70 else 1
    elif drug == "dopamine" and vasopressor_dose <= 5:
        scores["cardiovascular"] = 2
    elif drug == "dobutamine":
        scores["cardiovascular"] = 2
    elif (drug == "dopamine" and vasopressor_dose <= 15) or (
        drug in ("norepinephrine", "epinephrine") and vasopressor_dose <= 0.1
    ):
        scores["cardiovascular"] = 3
    else:
        scores["cardiovascular"] = 4

    # Neurological (GCS)
    if gcs == 15:
        scores["neurological"] = 0
    elif gcs >= 13:
        scores["neurological"] = 1
    elif gcs >= 10:
        scores["neurological"] = 2
    elif gcs >= 6:
        scores["neurological"] = 3
    else:
        scores["neurological"] = 4

    # Renal
    if creatinine < 1.2:
        scores["renal"] = 0
    elif creatinine <= 1.9:
        scores["renal"] = 1
    elif creatinine <= 3.4:
        scores["renal"] = 2
    elif creatinine <= 4.9 or urine_output_ml_per_day < 500:
        scores["renal"] = 3
    else:
        scores["renal"] = 4

    total = sum(scores.values())

    if total <= 1:
        mortality = "<10%"
    elif total <= 3:
        mortality = "~10%"
    elif total <= 6:
        mortality = "~20%"
    elif total <= 9:
        mortality = "~40%"
    elif total <= 11:
        mortality = "~50%"
    else:
        mortality = ">80%"

    return {
        "sofa_score": total,
        "max_score": 24,
        "organ_scores": scores,
        "predicted_mortality": mortality,
        "interpretation": f"SOFA {total}/24 → estimated mortality {mortality}",
    }


def calculate_apache_ii(
    age: int,
    rectal_temp_c: float,
    map_mmhg: float,
    heart_rate: int,
    respiratory_rate: int,
    fio2: float,
    pao2_mmhg: float,
    arterial_ph: float,
    serum_sodium: int,
    serum_potassium: float,
    serum_creatinine: float,
    hematocrit: float,
    wbc: float,
    gcs: int,
    acute_renal_failure: int,
    chronic_health_points: int,
) -> Dict:
    """
    Calculate APACHE II score.

    fio2: fraction inspired O2 (0.21-1.0)
    pao2_mmhg: arterial O2 partial pressure
    serum_creatinine: mg/dL
    hematocrit: percent (e.g. 42 for 42%)
    wbc: x10^3/µL
    acute_renal_failure: 1 if acute renal failure, 0 otherwise (doubles creatinine score)
    chronic_health_points: 2 if non-operative/emergency post-op, 5 if elective post-op, 0 otherwise
    """

    def aps_range(value, breakpoints_scores):
        """Find APS score for a given value using breakpoints."""
        for low, high, score in breakpoints_scores:
            if low <= value <= high:
                return score
        return 4  # default worst

    # Age points
    if age < 45:
        age_pts = 0
    elif age < 55:
        age_pts = 2
    elif age < 65:
        age_pts = 3
    elif age < 75:
        age_pts = 5
    else:
        age_pts = 6

    # Temperature
    temp_pts = aps_range(
        rectal_temp_c,
        [
            (41.0, 99, 4), (39.0, 40.9, 3), (38.5, 38.9, 1),
            (36.0, 38.4, 0), (34.0, 35.9, 1), (32.0, 33.9, 2),
            (30.0, 31.9, 3), (-99, 29.9, 4),
        ],
    )

    # MAP
    map_pts = aps_range(
        map_mmhg,
        [
            (160, 999, 4), (130, 159, 3), (110, 129, 2),
            (70, 109, 0), (50, 69, 2), (-99, 49, 4),
        ],
    )

    # Heart rate
    hr_pts = aps_range(
        heart_rate,
        [
            (180, 999, 4), (140, 179, 3), (110, 139, 2),
            (70, 109, 0), (55, 69, 2), (40, 54, 3), (-99, 39, 4),
        ],
    )

    # Respiratory rate
    rr_pts = aps_range(
        respiratory_rate,
        [
            (50, 999, 4), (35, 49, 3), (25, 34, 1),
            (12, 24, 0), (10, 11, 1), (6, 9, 2), (-99, 5, 4),
        ],
    )

    # Oxygenation
    if fio2 >= 0.5:
        # Use A-aDO2 proxy via PaO2 when FiO2 >= 0.5
        oxy_pts = aps_range(
            pao2_mmhg,
            [(500, 9999, 4), (350, 499, 3), (200, 349, 2), (-99, 199, 0)],
        )
    else:
        oxy_pts = aps_range(
            pao2_mmhg,
            [(70, 9999, 0), (61, 69, 1), (55, 60, 3), (-99, 54, 4)],
        )

    # pH
    ph_pts = aps_range(
        arterial_ph,
        [
            (7.7, 9.9, 4), (7.6, 7.69, 3), (7.5, 7.59, 1),
            (7.33, 7.49, 0), (7.25, 7.32, 2), (7.15, 7.24, 3), (-99, 7.14, 4),
        ],
    )

    # Sodium
    na_pts = aps_range(
        serum_sodium,
        [
            (180, 999, 4), (160, 179, 3), (155, 159, 2),
            (150, 154, 1), (130, 149, 0), (120, 129, 2),
            (111, 119, 3), (-99, 110, 4),
        ],
    )

    # Potassium
    k_pts = aps_range(
        serum_potassium,
        [
            (7.0, 99, 4), (6.0, 6.9, 3), (5.5, 5.9, 1),
            (3.5, 5.4, 0), (3.0, 3.4, 1), (2.5, 2.9, 2), (-99, 2.4, 4),
        ],
    )

    # Creatinine (doubled if acute renal failure)
    cr_pts = aps_range(
        serum_creatinine,
        [
            (3.5, 99, 4), (2.0, 3.4, 3), (1.5, 1.9, 2),
            (0.6, 1.4, 0), (-99, 0.59, 2),
        ],
    )
    if acute_renal_failure:
        cr_pts *= 2

    # Hematocrit
    hct_pts = aps_range(
        hematocrit,
        [
            (60, 999, 4), (50, 59.9, 2), (46, 49.9, 1),
            (30, 45.9, 0), (20, 29.9, 2), (-99, 19.9, 4),
        ],
    )

    # WBC
    wbc_pts = aps_range(
        wbc,
        [
            (40, 999, 4), (20, 39.9, 2), (15, 19.9, 1),
            (3, 14.9, 0), (1, 2.9, 2), (-99, 0.9, 4),
        ],
    )

    # GCS (15 - GCS)
    gcs_pts = 15 - gcs

    aps = (
        temp_pts + map_pts + hr_pts + rr_pts + oxy_pts + ph_pts
        + na_pts + k_pts + cr_pts + hct_pts + wbc_pts + gcs_pts
    )

    total = aps + age_pts + chronic_health_points

    # Approximate mortality lookup
    if total < 5:
        mortality = "~4%"
    elif total < 10:
        mortality = "~8%"
    elif total < 15:
        mortality = "~15%"
    elif total < 20:
        mortality = "~25%"
    elif total < 25:
        mortality = "~40%"
    elif total < 30:
        mortality = "~55%"
    elif total < 35:
        mortality = "~73%"
    else:
        mortality = "~85%"

    return {
        "apache_ii_score": total,
        "aps_score": aps,
        "age_points": age_pts,
        "chronic_health_points": chronic_health_points,
        "predicted_mortality": mortality,
        "interpretation": f"APACHE II {total} → estimated ICU mortality {mortality}",
    }


def calculate_gcs(eye_opening: int, verbal_response: int, motor_response: int) -> Dict:
    """
    Calculate Glasgow Coma Scale score.

    eye_opening: 1=None, 2=To pain, 3=To voice, 4=Spontaneous
    verbal_response: 1=None, 2=Sounds, 3=Words, 4=Confused, 5=Oriented
    motor_response: 1=None, 2=Extension, 3=Flexion, 4=Withdrawal, 5=Localises, 6=Obeys
    """
    eye_opening = max(1, min(4, eye_opening))
    verbal_response = max(1, min(5, verbal_response))
    motor_response = max(1, min(6, motor_response))

    total = eye_opening + verbal_response + motor_response

    if total >= 14:
        severity = "Mild TBI / minimal impairment"
    elif total >= 9:
        severity = "Moderate TBI"
    else:
        severity = "Severe TBI / coma"

    eye_labels = {1: "No eye opening", 2: "To pain", 3: "To voice", 4: "Spontaneous"}
    verbal_labels = {1: "No verbal", 2: "Incomprehensible sounds", 3: "Inappropriate words",
                     4: "Confused", 5: "Oriented"}
    motor_labels = {1: "No motor", 2: "Extension (decerebrate)", 3: "Abnormal flexion (decorticate)",
                    4: "Withdrawal", 5: "Localises pain", 6: "Obeys commands"}

    return {
        "gcs_total": total,
        "eye_opening": f"E{eye_opening} – {eye_labels[eye_opening]}",
        "verbal_response": f"V{verbal_response} – {verbal_labels[verbal_response]}",
        "motor_response": f"M{motor_response} – {motor_labels[motor_response]}",
        "severity_category": severity,
        "interpretation": f"GCS {total}/15 (E{eye_opening}V{verbal_response}M{motor_response}) – {severity}",
    }


def calculate_wells_dvt(
    active_cancer: int,
    paralysis_or_immobilisation: int,
    bedridden_or_recent_surgery: int,
    localised_tenderness: int,
    entire_leg_swollen: int,
    calf_swelling_3cm: int,
    pitting_oedema: int,
    collateral_veins: int,
    alternative_diagnosis_likely: int,
) -> Dict:
    """
    Calculate Wells DVT pretest probability score.

    All parameters are binary (0 or 1).
    alternative_diagnosis_likely: 1 if another diagnosis is at least as likely (-2 points).
    """
    components = {
        "active_cancer": ("Active cancer (treatment within 6 months)", active_cancer, 1),
        "paralysis": ("Paralysis/paresis or recent plaster immobilisation", paralysis_or_immobilisation, 1),
        "bedridden": ("Bedridden >3 days or major surgery <12 weeks", bedridden_or_recent_surgery, 1),
        "tenderness": ("Localised tenderness along deep venous system", localised_tenderness, 1),
        "entire_swollen": ("Entire leg swollen", entire_leg_swollen, 1),
        "calf_swelling": ("Calf swelling ≥3 cm compared with asymptomatic side", calf_swelling_3cm, 1),
        "pitting": ("Pitting oedema confined to symptomatic leg", pitting_oedema, 1),
        "collateral": ("Collateral superficial veins (non-varicose)", collateral_veins, 1),
        "alternative": ("Alternative diagnosis at least as likely", alternative_diagnosis_likely, -2),
    }

    score = 0
    detail = []
    for key, (label, val, pts) in components.items():
        contribution = val * pts
        score += contribution
        if val:
            detail.append(f"{label}: {'+' if pts > 0 else ''}{contribution}")

    if score <= 0:
        risk = "Low probability DVT (~3%)"
        action = "D-dimer test; if negative, DVT excluded"
    elif score <= 2:
        risk = "Moderate probability DVT (~17%)"
        action = "D-dimer or duplex ultrasound"
    else:
        risk = "High probability DVT (~75%)"
        action = "Duplex ultrasound immediately; treat empirically if unavailable"

    return {
        "wells_dvt_score": score,
        "probability_category": risk,
        "recommended_action": action,
        "scoring_detail": detail,
    }


def calculate_bmi(weight_kg: float, height_cm: float) -> Dict:
    """Calculate BMI and classify obesity category."""
    if height_cm <= 0 or weight_kg <= 0:
        return {"error": "Invalid height or weight values"}

    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m ** 2), 1)

    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25.0:
        category = "Normal weight"
    elif bmi < 30.0:
        category = "Overweight"
    elif bmi < 35.0:
        category = "Obese Class I"
    elif bmi < 40.0:
        category = "Obese Class II"
    else:
        category = "Obese Class III (morbidly obese)"

    return {
        "bmi": bmi,
        "category": category,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "interpretation": f"BMI {bmi} kg/m² – {category}",
    }


def convert_units(value: float, from_unit: str, to_unit: str) -> Dict:
    """Convert between common clinical units."""
    conversions: Dict[str, Dict[str, float]] = {
        # PaO2: mmHg ↔ kPa
        "mmhg_to_kpa": {"mmhg": 1 / 7.500617, "kpa": 7.500617},
        # Temperature
        "celsius_to_fahrenheit": {"celsius": 1.8, "offset_to": 32, "offset_from": -32},
        # Weight
        "kg_to_lbs": {"kg": 2.20462, "lbs": 0.453592},
        # Creatinine: mg/dL ↔ µmol/L
        "mgdl_to_umoll_cr": {"mg/dl": 88.42, "umol/l": 0.011312},
        # Bilirubin: mg/dL ↔ µmol/L
        "mgdl_to_umoll_bili": {"mg/dl_bili": 17.1, "umol/l_bili": 0.058480},
        # Glucose: mg/dL ↔ mmol/L
        "mgdl_to_mmoll_glucose": {"mg/dl_glucose": 0.05551, "mmol/l_glucose": 18.0182},
    }

    from_unit_lower = from_unit.lower().replace(" ", "")
    to_unit_lower = to_unit.lower().replace(" ", "")

    # Temperature special case
    if from_unit_lower in ("c", "celsius", "°c") and to_unit_lower in ("f", "fahrenheit", "°f"):
        result = value * 1.8 + 32
        return {"value": value, "from_unit": from_unit, "to_unit": to_unit,
                "result": round(result, 2), "formula": "°F = °C × 1.8 + 32"}

    if from_unit_lower in ("f", "fahrenheit", "°f") and to_unit_lower in ("c", "celsius", "°c"):
        result = (value - 32) / 1.8
        return {"value": value, "from_unit": from_unit, "to_unit": to_unit,
                "result": round(result, 2), "formula": "°C = (°F − 32) / 1.8"}

    # mmHg ↔ kPa
    if from_unit_lower in ("mmhg", "mm hg") and to_unit_lower in ("kpa",):
        return {"value": value, "from_unit": from_unit, "to_unit": to_unit,
                "result": round(value * 0.133322, 2), "formula": "kPa = mmHg × 0.1333"}

    if from_unit_lower in ("kpa",) and to_unit_lower in ("mmhg", "mm hg"):
        return {"value": value, "from_unit": from_unit, "to_unit": to_unit,
                "result": round(value * 7.50062, 2), "formula": "mmHg = kPa × 7.5006"}

    # Creatinine mg/dL ↔ µmol/L
    if from_unit_lower in ("mg/dl", "mg/dl_cr") and to_unit_lower in ("umol/l", "µmol/l"):
        return {"value": value, "from_unit": from_unit, "to_unit": to_unit,
                "result": round(value * 88.42, 1), "formula": "µmol/L = mg/dL × 88.42"}

    if from_unit_lower in ("umol/l", "µmol/l") and to_unit_lower in ("mg/dl", "mg/dl_cr"):
        return {"value": value, "from_unit": from_unit, "to_unit": to_unit,
                "result": round(value / 88.42, 3), "formula": "mg/dL = µmol/L / 88.42"}

    # Glucose mg/dL ↔ mmol/L
    if from_unit_lower in ("mg/dl_glucose",) and to_unit_lower in ("mmol/l",):
        return {"value": value, "from_unit": from_unit, "to_unit": to_unit,
                "result": round(value * 0.05551, 2), "formula": "mmol/L = mg/dL × 0.0555"}

    if from_unit_lower in ("mmol/l",) and to_unit_lower in ("mg/dl_glucose",):
        return {"value": value, "from_unit": from_unit, "to_unit": to_unit,
                "result": round(value * 18.0182, 1), "formula": "mg/dL = mmol/L × 18.0182"}

    return {
        "error": f"Unsupported conversion: {from_unit} → {to_unit}",
        "supported_conversions": [
            "Celsius ↔ Fahrenheit",
            "mmHg ↔ kPa",
            "mg/dL ↔ µmol/L (creatinine)",
            "mg/dL ↔ mmol/L (glucose)",
        ],
    }


def search_drug_interactions(drug1: str, drug2: str) -> Dict:
    """Look up known interactions between two drugs (mock clinical database)."""
    # Mock interaction database – representative examples for the evaluation scenarios
    interactions = {
        frozenset(["warfarin", "aspirin"]): {
            "severity": "Major",
            "description": "Increased bleeding risk. Combination significantly increases haemorrhagic events.",
            "management": "Avoid unless benefit clearly outweighs risk; use lowest effective doses; monitor INR closely.",
        },
        frozenset(["amiodarone", "warfarin"]): {
            "severity": "Major",
            "description": "Amiodarone inhibits CYP2C9 and CYP3A4, markedly increasing warfarin levels and INR.",
            "management": "Reduce warfarin dose by 30-50%; monitor INR weekly for at least 8 weeks.",
        },
        frozenset(["salbutamol", "ipratropium"]): {
            "severity": "None / Synergistic",
            "description": "No adverse interaction. These are commonly combined as additive bronchodilators.",
            "management": "Standard combination therapy; no dose adjustment required.",
        },
        frozenset(["prednisolone", "salbutamol"]): {
            "severity": "Minor",
            "description": "Prednisolone may enhance salbutamol-induced hypokalaemia.",
            "management": "Monitor potassium, especially in prolonged or high-dose use.",
        },
        frozenset(["amoxicillin", "prednisolone"]): {
            "severity": "Minor",
            "description": "Corticosteroids may slightly reduce antibiotic efficacy by suppressing immune response.",
            "management": "Generally safe to co-administer for AECOPD. Ensure adequate antibiotic dosing.",
        },
        frozenset(["metformin", "contrast"]): {
            "severity": "Moderate",
            "description": "Iodinated contrast agents can precipitate lactic acidosis in patients on metformin.",
            "management": "Hold metformin 48h before and after contrast; check renal function before restarting.",
        },
    }

    key = frozenset([drug1.lower(), drug2.lower()])
    if key in interactions:
        result = interactions[key]
        return {
            "drug1": drug1,
            "drug2": drug2,
            "interaction_found": True,
            **result,
        }

    return {
        "drug1": drug1,
        "drug2": drug2,
        "interaction_found": False,
        "severity": "None identified in database",
        "description": f"No clinically significant interaction found between {drug1} and {drug2}.",
        "management": "Proceed with standard prescribing; consult pharmacist for complex regimens.",
    }


def get_clinical_guideline(condition: str) -> Dict:
    """Retrieve a summary of clinical management guidelines for a given condition (mock database)."""
    guidelines = {
        "sepsis": {
            "guideline": "Surviving Sepsis Campaign 2021 / SSC Bundle",
            "key_points": [
                "Measure lactate; if >2 mmol/L, target <2 mmol/L",
                "Blood cultures x2 before antibiotics (do not delay >45 min)",
                "Broad-spectrum IV antibiotics within 1 hour of septic shock recognition",
                "30 mL/kg IV crystalloid bolus for hypotension or lactate ≥4",
                "Vasopressors (norepinephrine 1st line) to target MAP ≥65 mmHg",
                "Reassess fluid responsiveness; avoid fluid overload",
            ],
            "source": "Surviving Sepsis Campaign International Guidelines 2021",
        },
        "aecopd": {
            "guideline": "GOLD 2024 – Acute Exacerbation of COPD",
            "key_points": [
                "Controlled oxygen therapy: target SpO2 88-92%",
                "Short-acting bronchodilators (salbutamol + ipratropium) via nebuliser",
                "Oral prednisolone 40 mg for 5 days",
                "Antibiotics if purulent sputum or signs of pneumonia",
                "Consider NIV for persistent hypercapnic respiratory failure (pH 7.25-7.35)",
                "Intubation if NIV fails or pH <7.25",
            ],
            "source": "Global Initiative for Chronic Obstructive Lung Disease (GOLD) 2024",
        },
        "dvt": {
            "guideline": "NICE NG158 – Venous Thromboembolic Disease 2023",
            "key_points": [
                "Wells score to stratify pretest probability",
                "Low probability: D-dimer; if negative, VTE excluded",
                "High probability or positive D-dimer: proximal leg vein duplex ultrasound",
                "Anticoagulate with DOAC (apixaban or rivaroxaban) if DVT confirmed",
                "Duration: 3 months minimum; extend for unprovoked or high-risk cases",
            ],
            "source": "NICE NG158 (2023)",
        },
    }

    key = condition.lower().strip()
    for k, v in guidelines.items():
        if k in key or key in k:
            return {"condition": condition, **v}

    return {
        "condition": condition,
        "guideline": "Not found in local database",
        "key_points": ["Please consult UpToDate, BMJ Best Practice, or relevant specialty guidelines."],
        "source": "Local mock database",
    }


# ---------------------------------------------------------------------------
# Unified tool registry and schema definitions
# ---------------------------------------------------------------------------

TOOL_IMPLEMENTATIONS = {
    "calculate_qsofa": calculate_qsofa,
    "calculate_sofa": calculate_sofa,
    "calculate_apache_ii": calculate_apache_ii,
    "calculate_gcs": calculate_gcs,
    "calculate_wells_dvt": calculate_wells_dvt,
    "calculate_bmi": calculate_bmi,
    "convert_units": convert_units,
    "search_drug_interactions": search_drug_interactions,
    "get_clinical_guideline": get_clinical_guideline,
}


# OpenAI-compatible tool schema (also used as base for Gemini conversion)
TOOL_SCHEMAS: Dict[str, Dict] = {
    "calculate_qsofa": {
        "name": "calculate_qsofa",
        "description": "Calculate qSOFA score (0-3) to identify patients at high risk of sepsis. Score ≥2 indicates high risk.",
        "parameters": {
            "type": "object",
            "properties": {
                "respiratory_rate": {"type": "integer", "description": "Respiratory rate in breaths/min"},
                "altered_mentation": {"type": "integer", "description": "1 if GCS <15 or altered mental status, 0 if normal"},
                "systolic_bp": {"type": "integer", "description": "Systolic blood pressure in mmHg"},
            },
            "required": ["respiratory_rate", "altered_mentation", "systolic_bp"],
        },
    },
    "calculate_sofa": {
        "name": "calculate_sofa",
        "description": "Calculate SOFA score (0-24) to quantify organ failure severity and predict ICU mortality.",
        "parameters": {
            "type": "object",
            "properties": {
                "pf_ratio": {"type": "number", "description": "PaO2/FiO2 ratio (e.g. 300)"},
                "platelets": {"type": "integer", "description": "Platelet count x10^3/µL"},
                "bilirubin": {"type": "number", "description": "Total bilirubin mg/dL"},
                "map_mmhg": {"type": "number", "description": "Mean arterial pressure mmHg (use 0 if on vasopressors)"},
                "vasopressor_dose": {"type": "number", "description": "Vasopressor dose mcg/kg/min (0 if none)"},
                "vasopressor_drug": {"type": "string", "description": "Vasopressor name: none, dopamine, dobutamine, norepinephrine, epinephrine"},
                "gcs": {"type": "integer", "description": "Glasgow Coma Scale score 3-15"},
                "creatinine": {"type": "number", "description": "Serum creatinine mg/dL"},
                "urine_output_ml_per_day": {"type": "number", "description": "Urine output mL/day"},
            },
            "required": ["pf_ratio", "platelets", "bilirubin", "map_mmhg", "vasopressor_dose",
                         "vasopressor_drug", "gcs", "creatinine", "urine_output_ml_per_day"],
        },
    },
    "calculate_apache_ii": {
        "name": "calculate_apache_ii",
        "description": "Calculate APACHE II score (0-71) for ICU severity and mortality prediction.",
        "parameters": {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "description": "Patient age in years"},
                "rectal_temp_c": {"type": "number", "description": "Rectal temperature in Celsius"},
                "map_mmhg": {"type": "number", "description": "Mean arterial pressure mmHg"},
                "heart_rate": {"type": "integer", "description": "Heart rate bpm"},
                "respiratory_rate": {"type": "integer", "description": "Respiratory rate /min"},
                "fio2": {"type": "number", "description": "Fraction of inspired oxygen 0.21-1.0"},
                "pao2_mmhg": {"type": "number", "description": "Arterial PaO2 mmHg"},
                "arterial_ph": {"type": "number", "description": "Arterial blood gas pH"},
                "serum_sodium": {"type": "integer", "description": "Serum sodium mEq/L"},
                "serum_potassium": {"type": "number", "description": "Serum potassium mEq/L"},
                "serum_creatinine": {"type": "number", "description": "Serum creatinine mg/dL"},
                "hematocrit": {"type": "number", "description": "Hematocrit percent (e.g. 42 for 42%)"},
                "wbc": {"type": "number", "description": "White blood cell count x10^3/µL"},
                "gcs": {"type": "integer", "description": "Glasgow Coma Scale 3-15"},
                "acute_renal_failure": {"type": "integer", "description": "1 if acute renal failure, 0 if not"},
                "chronic_health_points": {"type": "integer", "description": "0, 2, or 5 based on operative status"},
            },
            "required": ["age", "rectal_temp_c", "map_mmhg", "heart_rate", "respiratory_rate",
                         "fio2", "pao2_mmhg", "arterial_ph", "serum_sodium", "serum_potassium",
                         "serum_creatinine", "hematocrit", "wbc", "gcs", "acute_renal_failure",
                         "chronic_health_points"],
        },
    },
    "calculate_gcs": {
        "name": "calculate_gcs",
        "description": "Calculate Glasgow Coma Scale (GCS) score 3-15 and classify severity of consciousness impairment.",
        "parameters": {
            "type": "object",
            "properties": {
                "eye_opening": {"type": "integer", "description": "Eye opening: 1=None, 2=To pain, 3=To voice, 4=Spontaneous"},
                "verbal_response": {"type": "integer", "description": "Verbal: 1=None, 2=Sounds, 3=Words, 4=Confused, 5=Oriented"},
                "motor_response": {"type": "integer", "description": "Motor: 1=None, 2=Extension, 3=Flexion, 4=Withdrawal, 5=Localises, 6=Obeys"},
            },
            "required": ["eye_opening", "verbal_response", "motor_response"],
        },
    },
    "calculate_wells_dvt": {
        "name": "calculate_wells_dvt",
        "description": "Calculate Wells DVT score to stratify pretest probability of deep vein thrombosis.",
        "parameters": {
            "type": "object",
            "properties": {
                "active_cancer": {"type": "integer", "description": "1 if active cancer, 0 if not"},
                "paralysis_or_immobilisation": {"type": "integer", "description": "1 if paralysis, paresis, or recent immobilisation"},
                "bedridden_or_recent_surgery": {"type": "integer", "description": "1 if bedridden >3 days or major surgery <12 weeks"},
                "localised_tenderness": {"type": "integer", "description": "1 if tenderness along deep venous system"},
                "entire_leg_swollen": {"type": "integer", "description": "1 if entire leg swollen"},
                "calf_swelling_3cm": {"type": "integer", "description": "1 if calf swelling ≥3 cm vs other side"},
                "pitting_oedema": {"type": "integer", "description": "1 if pitting oedema in symptomatic leg"},
                "collateral_veins": {"type": "integer", "description": "1 if collateral superficial veins"},
                "alternative_diagnosis_likely": {"type": "integer", "description": "1 if alternative diagnosis at least as likely (-2 points)"},
            },
            "required": ["active_cancer", "paralysis_or_immobilisation", "bedridden_or_recent_surgery",
                         "localised_tenderness", "entire_leg_swollen", "calf_swelling_3cm",
                         "pitting_oedema", "collateral_veins", "alternative_diagnosis_likely"],
        },
    },
    "calculate_bmi": {
        "name": "calculate_bmi",
        "description": "Calculate Body Mass Index (BMI) and classify obesity category.",
        "parameters": {
            "type": "object",
            "properties": {
                "weight_kg": {"type": "number", "description": "Body weight in kilograms"},
                "height_cm": {"type": "number", "description": "Height in centimetres"},
            },
            "required": ["weight_kg", "height_cm"],
        },
    },
    "convert_units": {
        "name": "convert_units",
        "description": "Convert between clinical measurement units (temperature, pressure, lab values).",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "Numeric value to convert"},
                "from_unit": {"type": "string", "description": "Source unit (e.g. mmHg, Celsius, mg/dL)"},
                "to_unit": {"type": "string", "description": "Target unit (e.g. kPa, Fahrenheit, µmol/L)"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
    "search_drug_interactions": {
        "name": "search_drug_interactions",
        "description": "Search for clinically significant interactions between two drugs.",
        "parameters": {
            "type": "object",
            "properties": {
                "drug1": {"type": "string", "description": "First drug name"},
                "drug2": {"type": "string", "description": "Second drug name"},
            },
            "required": ["drug1", "drug2"],
        },
    },
    "get_clinical_guideline": {
        "name": "get_clinical_guideline",
        "description": "Retrieve key management points from clinical guidelines for a given condition.",
        "parameters": {
            "type": "object",
            "properties": {
                "condition": {"type": "string", "description": "Medical condition (e.g. sepsis, AECOPD, DVT)"},
            },
            "required": ["condition"],
        },
    },
}


def get_openai_tools(tool_names: list) -> list:
    """Return tool definitions in OpenAI function-calling format for the given tool names."""
    return [
        {"type": "function", "function": TOOL_SCHEMAS[name]}
        for name in tool_names
        if name in TOOL_SCHEMAS
    ]


def get_gemini_tools(tool_names: list) -> list:
    """Return tool definitions as Gemini FunctionDeclaration-compatible dicts."""
    from google.genai import types

    declarations = []
    for name in tool_names:
        if name not in TOOL_SCHEMAS:
            continue
        schema = TOOL_SCHEMAS[name]
        declarations.append(
            types.FunctionDeclaration(
                name=schema["name"],
                description=schema["description"],
                parameters=schema["parameters"],
            )
        )
    return [types.Tool(function_declarations=declarations)]


def execute_tool(name: str, arguments: dict) -> Any:
    """Dispatch a tool call by name and return the result as a dict."""
    func = TOOL_IMPLEMENTATIONS.get(name)
    if func is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return func(**arguments)
    except Exception as exc:
        return {"error": f"Tool execution error: {exc}"}
