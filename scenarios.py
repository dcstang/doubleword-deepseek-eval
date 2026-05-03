"""
Five medical scenarios for the tool-calling evaluation.
Each scenario includes the clinical prompt, which tools we expect the model to use,
and a reference clinical answer for the judge to compare against.
"""

TOOL_CALLING_SCENARIOS = [
    {
        "id": 1,
        "title": "Septic Shock – Rapid Assessment",
        "prompt": (
            "A 65-year-old male presents to the ED with fever (38.9°C), acutely altered mental status "
            "(opens eyes to voice, confused speech, localises pain – E3V4M5), respiratory rate 24/min, "
            "heart rate 118 bpm, blood pressure 88/54 mmHg (MAP ~65 mmHg at best), SpO2 93% on 2L O2.\n\n"
            "Labs: WBC 18.2, Cr 2.4 mg/dL, bilirubin 2.1 mg/dL, platelets 98k, PaO2 74 mmHg on FiO2 0.28 "
            "(P/F ratio ~264). He is receiving norepinephrine at 0.15 mcg/kg/min. Urine output 120 mL over 24h.\n\n"
            "Please: (1) calculate his GCS, (2) calculate qSOFA, (3) calculate SOFA score, "
            "and (4) provide a summary of his severity and immediate management priorities."
        ),
        "available_tools": [
            "calculate_gcs", "calculate_qsofa", "calculate_sofa",
            "calculate_bmi", "get_clinical_guideline",
        ],
        "expected_tools": ["calculate_gcs", "calculate_qsofa", "calculate_sofa"],
        "reference_answer": (
            "GCS 12 (E3V4M5). qSOFA 3/3 (HIGH RISK). SOFA ~10-12 (high mortality ~40-50%). "
            "Diagnosis: septic shock. Immediate management: cultures, broad-spectrum antibiotics within 1h, "
            "fluid resuscitation 30mL/kg, vasopressor target MAP ≥65, lactate measurement, ICU admission."
        ),
    },
    {
        "id": 2,
        "title": "Post-Cardiac Surgery ICU Scoring",
        "prompt": (
            "72-year-old female admitted to ICU post-CABG x4 (off-pump). PMH: HTN, DM2, CKD stage 3.\n\n"
            "Current vitals: Temp 37.2°C (rectal), HR 88 SR, MAP 74 mmHg (on norepinephrine 0.08 mcg/kg/min), "
            "SpO2 98% (FiO2 0.5, PEEP 8), intubated.\n\n"
            "ABG: pH 7.32, PaCO2 42, PaO2 98 mmHg (P/F ratio = 196), HCO3 21.\n"
            "Labs: Na 138, K 4.2, Cr 1.8 mg/dL, bilirubin 1.0 mg/dL, Hct 34%, WBC 11.2, platelets 142k.\n"
            "GCS: E1V1M5 (deeply sedated).\n"
            "Urine output: 28 mL/hr (672 mL/day).\n\n"
            "Calculate her APACHE II score and SOFA score, then provide ICU prognostication and management priorities."
        ),
        "available_tools": [
            "calculate_apache_ii", "calculate_sofa", "calculate_gcs",
            "convert_units", "get_clinical_guideline",
        ],
        "expected_tools": ["calculate_apache_ii", "calculate_sofa"],
        "reference_answer": (
            "SOFA ~10 (respiratory 3, cardiovascular 3, renal 2, neurological 2) → ~40% predicted mortality. "
            "APACHE II ~22-26 → ~40-55% predicted mortality. "
            "Priorities: weaning sedation, lung-protective ventilation, renal monitoring, early physiotherapy."
        ),
    },
    {
        "id": 3,
        "title": "DVT Risk Stratification",
        "prompt": (
            "45-year-old male presents with 3-day history of right calf swelling, pain, and warmth. "
            "He returned from a 14-hour flight from Sydney to London 5 days ago. No recent surgery, "
            "immobilisation, or prior DVT/PE. No active cancer.\n\n"
            "Examination: right calf diameter 3.5 cm greater than left, pitting oedema to the knee, "
            "tenderness along the femoral vein distribution. No alternative diagnosis seems equally likely.\n\n"
            "Vital signs: HR 104, BP 126/78, RR 16, SpO2 97% RA, afebrile.\n"
            "Weight 87 kg, height 178 cm.\n\n"
            "Please: (1) calculate his BMI, (2) calculate his Wells DVT score, "
            "and (3) recommend the next diagnostic and treatment steps."
        ),
        "available_tools": [
            "calculate_bmi", "calculate_wells_dvt", "get_clinical_guideline",
            "search_drug_interactions", "convert_units",
        ],
        "expected_tools": ["calculate_bmi", "calculate_wells_dvt"],
        "reference_answer": (
            "BMI ~27.5 (overweight). Wells DVT score ~5-6 (high probability ~75%). "
            "Recommended action: immediate proximal leg vein duplex ultrasound; "
            "if confirmed, anticoagulate with DOAC (apixaban or rivaroxaban) for ≥3 months."
        ),
    },
    {
        "id": 4,
        "title": "Traumatic Brain Injury Assessment",
        "prompt": (
            "30-year-old male brought to trauma bay after high-speed MVC (not wearing seatbelt, ejected).\n\n"
            "On arrival: opens eyes to voice (E3), confused speech (V4), withdraws from pain (M4).\n"
            "Pupils: right 5 mm sluggish, left 3 mm reactive.\n"
            "Vital signs: BP 142/88, HR 96, RR 18, SpO2 96% RA.\n"
            "Weight 82 kg, height 181 cm.\n\n"
            "CT head: right temporal epidural haematoma 15 mm thick, midline shift 6 mm.\n\n"
            "Please: (1) calculate his GCS, (2) calculate his BMI, "
            "(3) assess neurological severity, and (4) outline immediate neurosurgical management."
        ),
        "available_tools": [
            "calculate_gcs", "calculate_bmi", "calculate_qsofa",
            "get_clinical_guideline", "convert_units",
        ],
        "expected_tools": ["calculate_gcs", "calculate_bmi"],
        "reference_answer": (
            "GCS 11 (E3V4M4) – moderate TBI. BMI ~25 (normal). "
            "Right EDH 15 mm with midline shift 6 mm and dilated pupil = surgical emergency. "
            "Immediate neurosurgical craniotomy / evacuation; maintain CPP >60 mmHg, avoid hypoxia and hypotension."
        ),
    },
    {
        "id": 5,
        "title": "AECOPD with Hypercapnic Respiratory Failure",
        "prompt": (
            "58-year-old female, 40 pack-year smoker, known COPD (GOLD stage 3), obesity. "
            "3-day worsening dyspnoea, increased purulent sputum.\n\n"
            "Vitals: Temp 38.1°C, HR 112, BP 148/92, RR 28, SpO2 82% on RA (improved to 90% on 28% Venturi).\n"
            "Weight: 95 kg, Height: 165 cm.\n\n"
            "ABG on 28% Venturi: pH 7.28, pCO2 68 mmHg, pO2 52 mmHg, HCO3 30, BE +2.\n"
            "CXR: hyperinflation, right lower lobe consolidation.\n\n"
            "Initial medications planned: salbutamol nebuliser, ipratropium nebuliser, prednisolone 40 mg PO, "
            "amoxicillin 500 mg TDS.\n\n"
            "Please: (1) calculate her BMI, (2) convert her pCO2 from mmHg to kPa, "
            "(3) check drug interactions for salbutamol + prednisolone and amoxicillin + prednisolone, "
            "(4) retrieve AECOPD guidelines, and (5) recommend ventilatory strategy."
        ),
        "available_tools": [
            "calculate_bmi", "convert_units", "search_drug_interactions",
            "get_clinical_guideline", "calculate_qsofa",
        ],
        "expected_tools": ["calculate_bmi", "convert_units", "search_drug_interactions", "get_clinical_guideline"],
        "reference_answer": (
            "BMI ~34.9 (Obese Class I). pCO2 68 mmHg ≈ 9.1 kPa. "
            "Salbutamol + prednisolone: minor interaction (hypokalaemia risk – monitor K+). "
            "Amoxicillin + prednisolone: minor, generally safe. "
            "AECOPD type 2 respiratory failure (pH 7.28, hypercapnia). "
            "NIV (BiPAP) indicated (pH 7.25-7.35, PaCO2 elevated). Target SpO2 88-92%. "
            "If NIV fails or pH <7.25, consider intubation."
        ),
    },
]
