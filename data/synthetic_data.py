import pandas as pd
import numpy as np
import os

np.random.seed(42)
os.makedirs("data/synthetic", exist_ok=True)

# ── 1. STUDENT ───────────────────────────────────────────────
def generate_student(n=1000):
    print("=== Generating Student Synthetic Data ===")
    records = []

    for _ in range(n):
        case_type = np.random.choice(
            ["easy_pass", "easy_fail", "borderline", "adversarial"],
            p=[0.25, 0.25, 0.35, 0.15]
        )

        if case_type == "easy_pass":
            r = {
                "school": np.random.randint(0, 2),
                "sex": np.random.randint(0, 2),
                "age": np.random.randint(15, 19),
                "studytime": np.random.randint(3, 5),
                "failures": 0,
                "schoolsup": 0,
                "famsup": np.random.randint(0, 2),
                "paid": np.random.randint(0, 2),
                "activities": 1,
                "higher": 1,
                "internet": 1,
                "romantic": 0,
                "famrel": np.random.randint(3, 5),
                "freetime": np.random.randint(2, 4),
                "goout": np.random.randint(1, 3),
                "dalc": 1,
                "walc": np.random.randint(1, 2),
                "health": np.random.randint(3, 5),
                "absences": np.random.randint(0, 5),
                "g1": np.random.randint(12, 18),
                "g2": np.random.randint(13, 19),
                "pass": 1,
                "synthetic": 1
            }
        elif case_type == "easy_fail":
            r = {
                "school": np.random.randint(0, 2),
                "sex": np.random.randint(0, 2),
                "age": np.random.randint(16, 22),
                "studytime": np.random.randint(1, 2),
                "failures": np.random.randint(2, 4),
                "schoolsup": 1,
                "famsup": 0,
                "paid": 0,
                "activities": 0,
                "higher": 0,
                "internet": np.random.randint(0, 2),
                "romantic": 1,
                "famrel": np.random.randint(1, 3),
                "freetime": np.random.randint(3, 5),
                "goout": np.random.randint(3, 5),
                "dalc": np.random.randint(3, 5),
                "walc": np.random.randint(3, 5),
                "health": np.random.randint(1, 3),
                "absences": np.random.randint(10, 30),
                "g1": np.random.randint(3, 8),
                "g2": np.random.randint(3, 8),
                "pass": 0,
                "synthetic": 1
            }
        elif case_type == "borderline":
            r = {
                "school": np.random.randint(0, 2),
                "sex": np.random.randint(0, 2),
                "age": np.random.randint(16, 19),
                "studytime": np.random.randint(2, 3),
                "failures": np.random.randint(0, 2),
                "schoolsup": np.random.randint(0, 2),
                "famsup": np.random.randint(0, 2),
                "paid": np.random.randint(0, 2),
                "activities": np.random.randint(0, 2),
                "higher": np.random.randint(0, 2),
                "internet": np.random.randint(0, 2),
                "romantic": np.random.randint(0, 2),
                "famrel": np.random.randint(2, 4),
                "freetime": np.random.randint(2, 4),
                "goout": np.random.randint(2, 4),
                "dalc": np.random.randint(1, 3),
                "walc": np.random.randint(1, 3),
                "health": np.random.randint(2, 4),
                "absences": np.random.randint(3, 12),
                "g1": np.random.randint(8, 13),
                "g2": np.random.randint(8, 13),
                "pass": np.random.randint(0, 2),
                "synthetic": 1
            }
        else:  # adversarial
            r = {
                "school": np.random.randint(0, 2),
                "sex": np.random.randint(0, 2),
                "age": np.random.randint(15, 22),
                "studytime": np.random.randint(3, 5),
                "failures": np.random.randint(2, 4),
                "schoolsup": np.random.randint(0, 2),
                "famsup": np.random.randint(0, 2),
                "paid": np.random.randint(0, 2),
                "activities": np.random.randint(0, 2),
                "higher": 1,
                "internet": 1,
                "romantic": np.random.randint(0, 2),
                "famrel": np.random.randint(1, 5),
                "freetime": np.random.randint(1, 5),
                "goout": np.random.randint(1, 5),
                "dalc": np.random.randint(1, 5),
                "walc": np.random.randint(1, 5),
                "health": np.random.randint(1, 5),
                "absences": np.random.randint(0, 30),
                "g1": np.random.randint(3, 18),
                "g2": np.random.randint(3, 18),
                "pass": np.random.randint(0, 2),
                "synthetic": 1
            }
        records.append(r)

    df = pd.DataFrame(records)
    df.to_csv("data/synthetic/student_synthetic.csv", index=False)
    print(f"Generated: {len(df)} rows | Pass rate: {df['pass'].mean():.1%}")
    print(f"Saved: data/synthetic/student_synthetic.csv\n")
    return df

# ── 2. HR ────────────────────────────────────────────────────
def generate_hr(n=1000):
    print("=== Generating HR Synthetic Data ===")
    records = []

    for _ in range(n):
        case_type = np.random.choice(
            ["easy_stay", "easy_leave", "borderline", "adversarial"],
            p=[0.25, 0.25, 0.35, 0.15]
        )

        if case_type == "easy_stay":
            r = {
                "age": np.random.randint(30, 50),
                "dailyrate": np.random.randint(800, 1400),
                "distancefromhome": np.random.randint(1, 5),
                "education": np.random.randint(3, 5),
                "environmentsatisfaction": np.random.randint(3, 5),
                "hourlyrate": np.random.randint(60, 100),
                "jobinvolvement": np.random.randint(3, 4),
                "joblevel": np.random.randint(2, 5),
                "jobsatisfaction": np.random.randint(3, 5),
                "monthlyincome": np.random.randint(5000, 15000),
                "numcompaniesworked": np.random.randint(1, 3),
                "overtime": 0,
                "percentsalaryhike": np.random.randint(15, 25),
                "performancerating": np.random.randint(3, 4),
                "relationshipsatisfaction": np.random.randint(3, 5),
                "stockoptionlevel": np.random.randint(1, 3),
                "totalworkingyears": np.random.randint(8, 25),
                "worklifebalance": np.random.randint(3, 4),
                "yearsatcompany": np.random.randint(5, 20),
                "attrition": 0,
                "synthetic": 1
            }
        elif case_type == "easy_leave":
            r = {
                "age": np.random.randint(22, 32),
                "dailyrate": np.random.randint(200, 600),
                "distancefromhome": np.random.randint(15, 30),
                "education": np.random.randint(1, 3),
                "environmentsatisfaction": np.random.randint(1, 2),
                "hourlyrate": np.random.randint(30, 55),
                "jobinvolvement": np.random.randint(1, 2),
                "joblevel": 1,
                "jobsatisfaction": np.random.randint(1, 2),
                "monthlyincome": np.random.randint(1000, 3000),
                "numcompaniesworked": np.random.randint(4, 9),
                "overtime": 1,
                "percentsalaryhike": np.random.randint(11, 14),
                "performancerating": np.random.randint(3, 4),
                "relationshipsatisfaction": np.random.randint(1, 2),
                "stockoptionlevel": 0,
                "totalworkingyears": np.random.randint(1, 5),
                "worklifebalance": np.random.randint(1, 2),
                "yearsatcompany": np.random.randint(0, 3),
                "attrition": 1,
                "synthetic": 1
            }
        else:
            r = {
                "age": np.random.randint(22, 55),
                "dailyrate": np.random.randint(200, 1400),
                "distancefromhome": np.random.randint(1, 30),
                "education": np.random.randint(1, 5),
                "environmentsatisfaction": np.random.randint(1, 5),
                "hourlyrate": np.random.randint(30, 100),
                "jobinvolvement": np.random.randint(1, 4),
                "joblevel": np.random.randint(1, 5),
                "jobsatisfaction": np.random.randint(1, 5),
                "monthlyincome": np.random.randint(1000, 15000),
                "numcompaniesworked": np.random.randint(0, 9),
                "overtime": np.random.randint(0, 2),
                "percentsalaryhike": np.random.randint(11, 25),
                "performancerating": np.random.randint(3, 4),
                "relationshipsatisfaction": np.random.randint(1, 5),
                "stockoptionlevel": np.random.randint(0, 3),
                "totalworkingyears": np.random.randint(0, 30),
                "worklifebalance": np.random.randint(1, 4),
                "yearsatcompany": np.random.randint(0, 25),
                "attrition": np.random.randint(0, 2),
                "synthetic": 1
            }
        records.append(r)

    df = pd.DataFrame(records)
    df.to_csv("data/synthetic/hr_synthetic.csv", index=False)
    print(f"Generated: {len(df)} rows | Attrition rate: {df['attrition'].mean():.1%}")
    print(f"Saved: data/synthetic/hr_synthetic.csv\n")
    return df

# ── 3. BEHAVIORAL ─────────────────────────────────────────────
def generate_behavioral(n=1000):
    print("=== Generating Behavioral Synthetic Data ===")

    genuine_templates = [
        "Stayed here for a conference. Room was small but functional. Shower pressure was weak.",
        "The hotel was conveniently located near the airport. Breakfast had limited options.",
        "Check-in was slow but staff apologized. Room had a broken lamp which they fixed promptly.",
        "Decent stay overall. The gym equipment was outdated. Bed was comfortable enough.",
        "Parking was a nightmare but the room itself was clean and quiet. Would consider again.",
        "Nothing special about this place. Gets the job done for a business trip.",
        "Had a minor issue with noise from the hallway. Front desk resolved it quickly.",
        "The view from room 412 was disappointing — just a parking lot. Location was good though.",
        "Wifi dropped twice during my stay. Otherwise a standard hotel experience.",
        "Room service took 45 minutes. Food was warm, portions reasonable.",
    ]

    deceptive_templates = [
        "ABSOLUTELY LOVE THIS HOTEL! Best stay of my life! Will come back every year without fail!",
        "Perfect perfect perfect! Everything was flawless! Staff were angels! Zero complaints!",
        "This hotel changed my life! Most luxurious experience imaginable! Worth every penny!",
        "Cannot recommend enough!! The best hotel in the entire city by far! Simply outstanding!",
        "WOW WOW WOW! Exceeded every expectation! Magical from check-in to check-out!",
        "Five stars is not enough! This place deserves ten stars! Absolutely phenomenal service!",
        "Best hotel experience ever!! Staff were incredible! Rooms were pristine! Just amazing!",
        "Unbelievable value! Stunning rooms! Exceptional staff! Will tell everyone about this gem!",
        "Dreams do come true and this hotel is proof! Spectacular in every possible way!",
        "A masterpiece of hospitality! Every detail perfect! Management deserves highest praise!",
    ]

    records = []
    hotels = ["Grand Hotel", "City Inn", "Plaza Hotel", "Royal Suites",
              "Lake View", "Metro Lodge", "Harbor Inn", "Peak Hotel"]
    sources = ["tripadvisor", "expedia", "booking", "google", "yelp"]

    for _ in range(n):
        is_deceptive = np.random.randint(0, 2)
        hotel = np.random.choice(hotels)
        template = np.random.choice(
            deceptive_templates if is_deceptive else genuine_templates
        )
        text = template.replace("this hotel", hotel).replace("This hotel", hotel)

        records.append({
            "text": text,
            "deceptive": is_deceptive,
            "polarity": 1 if is_deceptive else np.random.randint(0, 2),
            "hotel": np.random.randint(0, len(hotels)),
            "source": np.random.randint(0, len(sources)),
            "word_count": len(text.split()),
            "exclamation_count": text.count("!"),
            "has_superlative": int(any(w in text.lower() for w in
                ["best", "amazing", "perfect", "incredible", "outstanding",
                 "flawless", "phenomenal", "spectacular", "masterpiece"])),
            "synthetic": 1
        })

    df = pd.DataFrame(records)
    df.to_csv("data/synthetic/behavioral_synthetic.csv", index=False)
    print(f"Generated: {len(df)} rows | Deceptive rate: {df['deceptive'].mean():.1%}")
    print(f"Saved: data/synthetic/behavioral_synthetic.csv\n")
    return df

# ── RUN ALL ───────────────────────────────────────────────────
if __name__ == "__main__":
    generate_student(1000)
    generate_hr(1000)
    generate_behavioral(1000)
    print("=== Phase 2 Stage 2.3 COMPLETE ===")