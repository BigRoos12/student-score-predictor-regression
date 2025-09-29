
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

#Reading and examining raw data
csv = pd.read_csv("data.csv")
print(csv.head())
print("-"*60)
print(csv.info())
print("-"*60)
print(csv.describe())

temporary_data = csv[["study_hours_per_week", "sleep_hours_per_day", "final_grade"]]
#mean
means = temporary_data.mean()
print("میانگین‌ها:")
print(means)

#median
medians = temporary_data.median()
print("میانه ها:")
print(medians)
#Q1
print(f"Q1 مطالعه: {temporary_data['study_hours_per_week'].quantile(0.25):.2f}")
print(f"Q1 خواب: {temporary_data['sleep_hours_per_day'].quantile(0.25):.2f}")
print(f"Q1 نمره: {temporary_data['final_grade'].quantile(0.25):.2f}")


#create a piechart for participation_level
csv = csv [["study_hours_per_week","sleep_hours_per_day","attendance_percentage","assignments_completed","final_grade","participation_level","internet_access","parental_education","extracurricular","part_time_job"]]
csv['participation_level'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("participation_level in class:")
plt.ylabel('')
plt.show()


#check dataset for missing
print("Checking for missing values in the original dataset")
print("=" * 50)

missing_summary = csv.isnull().sum()
print("Number of missing values in each column:")
print(missing_summary)

print(f"\n Total sum of missing values: {csv.isnull().sum().sum()}")
print(f"percentage of missing values: {(csv.isnull().sum().sum() / (len(csv) * len(csv.columns)) * 100):.2f}%")

columns_with_missing = missing_summary[missing_summary > 0]
if len(columns_with_missing) > 0:
    print(f"\n rows with missing values: {list(columns_with_missing.index)}")
else:
    print("\n dataset dont have any missing value")

csv['parental_education'].fillna('Unknown', inplace=True)
print(f"new missing values: {csv['parental_education'].isnull().sum()}")

csv.loc[csv['sleep_hours_per_day'] < 2, 'sleep_hours_per_day'] = 2
csv.loc[csv['sleep_hours_per_day'] > 14, 'sleep_hours_per_day'] = 14

# مطالعه: مقادیر کمتر از ۰.۳۰ رو با ۰.۳۰، بیشتر از ۱۰۰ رو با ۱۰۰ جایگزین کن
csv.loc[csv['study_hours_per_week'] < 0.30, 'study_hours_per_week'] = 0.30
csv.loc[csv['study_hours_per_week'] > 100, 'study_hours_per_week'] = 100

# حضور: مقادیر کمتر از ۰ رو با ۰، بیشتر از ۱ رو با ۱ جایگزین کن
csv.loc[csv['attendance_percentage'] < 0, 'attendance_percentage'] = 0
csv.loc[csv['attendance_percentage'] > 1, 'attendance_percentage'] = 1

# نمره: مقادیر کمتر از ۰ رو با ۰، بیشتر از ۱۰۰ رو با ۱۰۰ جایگزین کن
csv.loc[csv['final_grade'] < 0, 'final_grade'] = 0
csv.loc[csv['final_grade'] > 100, 'final_grade'] = 100

#Sleeping more than 14 hours or less than 2
print(csv[(csv['sleep_hours_per_day'] < 2) | (csv['sleep_hours_per_day'] > 14)])
#study more than 100 hours or less than 0.30
print(csv[(csv['study_hours_per_week'] < 0.30) | (csv['study_hours_per_week'] > 100)])
#attendance_outliers more than 1 or less than 0
print(csv[(csv['attendance_percentage'] < 0) | (csv['attendance_percentage'] > 1)])
#final grade more than 100 or less than 0
print(csv[(csv['final_grade'] < 0) | (csv['final_grade'] > 100)])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# study
sns.histplot(data=csv, x='study_hours_per_week', kde=True, ax=axes[0],bins = 20)
axes[0].set_title('study_hours_per_week', fontsize=16)

# ۲. خواب
sns.histplot(data=csv, x='sleep_hours_per_day', kde=True, ax=axes[1], color='orange',bins = 20)
axes[1].set_title('sleep_hours_per_day', fontsize=16)

# ۳. نمره
sns.histplot(data=csv, x='final_grade', kde=True, ax=axes[2], color='green',bins = 20)
axes[2].set_title('final_grade', fontsize=16)

plt.tight_layout()
plt.show()

# باکس پلات‌های زیباتر با seaborn
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.boxplot(y=csv['study_hours_per_week'], ax=axes[0], color='blue')
axes[0].set_title('study_hours_per_week')

sns.boxplot(y=csv['sleep_hours_per_day'], ax=axes[1], color='green')
axes[1].set_title('sleep_hours_per_day')

sns.boxplot(y=csv['final_grade'], ax=axes[2], color='coral')
axes[2].set_title('final_grade')

plt.tight_layout()
plt.show()

# مرحله 4: ساخت ماتریس همبستگی
corr_matrix = csv[["study_hours_per_week", "sleep_hours_per_day","attendance_percentage", "final_grade"]].corr()

# مرحله 5: رسم هیت مپ
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap="gist_heat", fmt=".2f",linewidths=0.10,linecolor='black')
plt.title("Correlation Heatmap", fontsize=14)
plt.show()


csv['grade_class'] = np.where(csv['final_grade'] < 70, "unfavored", "favored")
print(csv['grade_class'].unique())

numeric_features = csv[["study_hours_per_week","sleep_hours_per_day","attendance_percentage","assignments_completed","participation_level","final_grade"]]

for feature in numeric_features:
    plt.figure(figsize=(8,4))
    sns.histplot(data=csv, x=feature, hue='grade_class', kde=False, bins=15, alpha=0.6)
    plt.title(f"Distribution of {feature} by Class")
    plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(data=csv, x='study_hours_per_week', y='final_grade', hue='grade_class', s=80, alpha=0.7)
plt.title("Study Hours vs Final Grade")
plt.xlabel("Study Hours per Week")
plt.ylabel("Final Grade")
plt.legend(title='Class')
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(data=csv, x='attendance_percentage', y='final_grade', hue='grade_class', s=80, alpha=0.7)
plt.title("Attendance vs Final Grade")
plt.xlabel("Attendance Percentage")
plt.ylabel("Final Grade")
plt.legend(title='Class')
plt.show()


plt.figure(figsize=(6,4))
sns.scatterplot(data=csv, x='assignments_completed', y='final_grade', hue='grade_class', s=80, alpha=0.7)
plt.title("Assignments Completed vs Final Grade")
plt.xlabel("Assignments Completed")
plt.ylabel("Final Grade")
plt.legend(title='Class')
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(data=csv,x='sleep_hours_per_day', y='final_grade', hue='grade_class', s=80, alpha=0.7)
plt.title("sleep_hours_per_day vs Final Grade")
plt.xlabel("Assignments Completed")
plt.ylabel("Final Grade")
plt.legend(title='Class')
plt.show()

study_hours_per_week = float(input("study_hours_per_week: "))
attendance_percentage = float(input("attendance_percentage: "))
assignments_completed = float(input("assignments_completed: "))
participation_level = input("participation_level (high, medium, low): ").lower()
internet_access = input("internet_access (yes or no): ").lower()
parental_education = input("parental_education (high school, master, PhD): ").lower()
part_time_job = input("part_time_job (yes or no): ").lower()
extracurricular = input("extracurricular (yes or no): ").lower()
sleep_hours = float(input("sleep_hours per night: "))

score = 0
score += min(study_hours_per_week * 1.5, 30)
score += attendance_percentage * 0.5
score += min(assignments_completed * 0.7, 25)

if participation_level == "high":
    score += 10
elif participation_level == "medium":
    score += 5

if internet_access == "yes":
    score += 5

if parental_education == "master":
    score += 3
elif parental_education == "phd":
    score += 5

if part_time_job == "yes":
    score -= 5

if extracurricular == "yes":
    score += 5

if 7 <= sleep_hours <= 9:
    score += 10
elif sleep_hours < 5:
    score -= 10
elif sleep_hours > 10:
    score -= 5

score = max(0, min(score, 100))
label = "favored" if score >= 70 else "unfavored"

print(f"Predicted final score: {score}")
print(f"Predicted class: {label}")



preds = []
for i in range(len(csv)):
    row = csv.iloc[i]
    if row["study_hours_per_week"] > 15 and row["attendance_percentage"] > 80:
        pred = 90
    elif row["assignments_completed"] > 5 and row["sleep_hours_per_day"] >= 7:
        pred = 75
    else:
        pred = 60
    preds.append(pred)

csv["predicted_grade"] = preds

csv['error'] = csv['predicted_grade'] - csv['final_grade']

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    csv['final_grade'],
    csv['predicted_grade'],
    c=csv['error'],          # رنگ بر اساس خطا
    cmap='coolwarm',
    alpha=0.7,
    edgecolor='k'
)

plt.plot([0, 100], [0, 100], color='black', linestyle='--', linewidth=1.5, label="Ideal Line")
plt.colorbar(scatter, label="Prediction Error")
plt.xlabel("Actual Final Grade")
plt.ylabel("Predicted grade")
plt.title("Actual vs Predicted Final Grades")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

categorical_features = ["participation_level","internet_access","parental_education","extracurricular","part_time_job"]
numeric_features = ["study_hours_per_week","sleep_hours_per_day","attendance_percentage","assignments_completed"]
csv_encoded = csv.copy()
label_encoders = {}

# --- Encode categorical features ---
for col in categorical_features:
    le = LabelEncoder()
    csv_encoded[col] = le.fit_transform(csv_encoded[col].astype(str))
    label_encoders[col] = le

# --- Scale numeric features ---
scaler = StandardScaler()
csv_encoded[numeric_features] = scaler.fit_transform(csv_encoded[numeric_features])

# --- آماده سازی داده‌ها ---
features = numeric_features + categorical_features
X = csv_encoded[features]
y = csv_encoded['final_grade']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- آموزش مدل ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- پیش‌بینی و ارزیابی ---
y_pred = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# --- ضرایب مدل ---
print("\n تاثیر هر ویژگی بر نمره نهایی:")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:+.3f}")

