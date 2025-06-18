<div align="center">

# 🇰🇿 KZgovData: Kazakhstan Government Complaints Dataset

<p align="center">
  <img src="https://img.shields.io/badge/Dataset-Real%20World-blue?style=for-the-badge&logo=database" alt="Dataset">
  <img src="https://img.shields.io/badge/Languages-Kazakh%20%7C%20Russian-green?style=for-the-badge&logo=google-translate" alt="Languages">
  <img src="https://img.shields.io/badge/Size-256KB-orange?style=for-the-badge&logo=files" alt="Size">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/AdilzhanB/KZgovData?style=social" alt="Stars">
  <img src="https://img.shields.io/github/forks/AdilzhanB/KZgovData?style=social" alt="Forks">
  <img src="https://img.shields.io/github/watchers/AdilzhanB/KZgovData?style=social" alt="Watchers">
</p>

**Bilingual dataset of Kazakhstani citizen complaints to local government (akimats) with official responses in Kazakh and Russian languages**

[🚀 Quick Start](#-quick-start) • [📊 Dataset Overview](#-dataset-overview) • [💾 Download](#-download) • [🔧 Usage](#-usage) • [📈 Applications](#-applications) • [🤝 Contributing](#-contributing)

</div>

---

## 🌟 Overview

**KZgovData** is a comprehensive bilingual dataset containing real citizen complaints submitted to Kazakhstan local governments (akimats) along with official responses. Each record includes both Kazakh and Russian translations, making it ideal for training multilingual government communication systems and studying citizen-government interactions in Kazakhstan.

<div align="center">

### 🎯 Key Features

| Feature | Description |
|---------|-------------|
| 🌐 **Bilingual** | Every complaint in both Kazakh (kz) and Russian (ru) |
| 🏛️ **Government Official** | Real complaints to Kazakhstan akimats |
| 📝 **Complete Records** | Complaints + official government responses |
| 🗂️ **Categorized** | Organized by government service categories |
| 📍 **Regional** | Covers different regions of Kazakhstan |
| 🚨 **Urgency Levels** | Classified by complaint urgency |
| 📊 **Compact** | Efficient 256KB dataset size |

</div>

---

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📊 Dataset Overview](#-dataset-overview)
- [🏗️ Data Structure](#️-data-structure)
- [💾 Download](#-download)
- [🔧 Usage](#-usage)
- [📈 Applications](#-applications)
- [📚 Examples](#-examples)
- [⚠️ Limitations](#️-limitations)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install required packages
pip install pandas numpy matplotlib seaborn plotly
```

### Loading the Dataset

```python
import pandas as pd
import json

# Load from CSV
df = pd.read_csv("kzgov_complaints.csv")

# Or load from JSON
with open("kzgov_complaints.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    df = pd.DataFrame(data)

print(f"Dataset size: {len(df)} records")
print(f"Dataset columns: {list(df.columns)}")
```

### Basic Usage

```python
# View sample record
sample = df.iloc[0]
print("Sample Complaint:")
print(f"ID: {sample['id']}")
print(f"Kazakh: {sample['text_kz']}")
print(f"Russian: {sample['text_ru']}")
print(f"Category: {sample['category']}")
print(f"Region: {sample['region']}")
print(f"Response: {sample['reply_text']}")

# Filter by category
zhkh_complaints = df[df['category'] == 'ЖКХ']
print(f"ЖКХ complaints: {len(zhkh_complaints)}")

# Filter by urgency
urgent_complaints = df[df['urgency'] == 'высокая']
print(f"High urgency complaints: {len(urgent_complaints)}")
```

---

## 🏗️ Data Structure

### 📝 Schema Definition

```python
{
    "id": int,               # Unique complaint identifier
    "text_kz": str,          # Complaint text in Kazakh language
    "text_ru": str,          # Complaint text in Russian language  
    "category": str,         # Government service category
    "urgency": str,          # Urgency level of the complaint
    "region": str,           # Kazakhstan region/oblast
    "status": str,           # Current complaint status
    "reply_text": str,       # Official government response
    "duplicate": bool        # Whether this is a duplicate complaint
}
```

### 🎯 Field Descriptions

<div align="center">

| Field | Type | Description | Example Values |
|-------|------|-------------|----------------|
| `id` | Integer | Unique complaint identifier | 1, 2, 3, ... |
| `text_kz` | String | Complaint in Kazakh | "Құрметті әкімдік, біздің ауылда жарық жоқ." |
| `text_ru` | String | Complaint in Russian | "Уважаемая акимат, в нашем селе нет света." |
| `category` | String | Service category | "ЖКХ", "Образование", "Здравоохранение" |
| `urgency` | String | Priority level | "высокая", "средняя", "низкая" |
| `region` | String | Administrative region | "Туркестанская область", "Алматинская область" |
| `status` | String | Processing status | "новая", "в обработке", "решена" |
| `reply_text` | String | Government response | "Проблема передана в отдел энергетики..." |
| `duplicate` | Boolean | Duplicate flag | true, false |

</div>

---

## 📊 Dataset Overview

### 📈 Dataset Statistics

<div align="center">

| Metric | Value |
|--------|-------|
| **Total Records** | Variable (estimated 100-500 records) |
| **File Size** | 256 KB |
| **Languages** | Kazakh (kk), Russian (ru) |
| **Format** | CSV, JSON |
| **Encoding** | UTF-8 |
| **Date Created** | June 18, 2025 |

</div>

### 🗂️ Categories Analysis

```python
# Analyze categories
category_counts = df['category'].value_counts()
print("Complaint Categories:")
for category, count in category_counts.items():
    print(f"  {category}: {count} complaints")

# Analyze urgency levels
urgency_counts = df['urgency'].value_counts()
print("\nUrgency Levels:")
for urgency, count in urgency_counts.items():
    print(f"  {urgency}: {count} complaints")

# Analyze regions
region_counts = df['region'].value_counts()
print("\nRegions:")
for region, count in region_counts.items():
    print(f"  {region}: {count} complaints")
```

### 🏷️ Common Categories

<div align="center">

| Category (Russian) | Category (English) | Description |
|-------------------|-------------------|-------------|
| **ЖКХ** | Utilities & Housing | Water, electricity, heating, housing services |
| **Образование** | Education | Schools, kindergartens, educational services |
| **Здравоохранение** | Healthcare | Medical services, hospitals, clinics |
| **Дороги** | Roads | Road maintenance, traffic, transportation |
| **Социальные услуги** | Social Services | Benefits, social support, assistance |
| **Экология** | Environment | Waste management, pollution, environmental issues |
| **Административные** | Administrative | Documents, permits, bureaucratic services |

</div>

---

## 💾 Download

### 🔗 Available Formats

<div align="center">

| Format | Description | Best For |
|--------|-------------|----------|
| **CSV** | Comma-separated values | Data analysis, Excel |
| **JSON** | JavaScript Object Notation | Web applications, APIs |
| **Parquet** | Columnar storage | Big data processing |

</div>

### 📥 Download Methods

<details>
<summary><strong>Method 1: Direct Download</strong></summary>

```bash
# Clone the repository
git clone https://github.com/AdilzhanB/KZgovData.git
cd KZgovData

# Files available:
# - data/kzgov_complaints.csv
# - data/kzgov_complaints.json
# - data/kzgov_complaints.xlsx
```

</details>

<details>
<summary><strong>Method 2: Python Download</strong></summary>

```python
import requests
import pandas as pd

# Download CSV directly
url = "https://raw.githubusercontent.com/AdilzhanB/KZgovData/main/data/kzgov_complaints.csv"
df = pd.read_csv(url)

# Download JSON
json_url = "https://raw.githubusercontent.com/AdilzhanB/KZgovData/main/data/kzgov_complaints.json"
response = requests.get(json_url)
data = response.json()
df = pd.DataFrame(data)
```

</details>

<details>
<summary><strong>Method 3: wget/curl</strong></summary>

```bash
# Using wget
wget https://github.com/AdilzhanB/KZgovData/raw/main/data/kzgov_complaints.csv

# Using curl
curl -O https://github.com/AdilzhanB/KZgovData/raw/main/data/kzgov_complaints.json
```

</details>

---

## 🔧 Usage

### 🤖 Text Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load data
df = pd.read_csv("kzgov_complaints.csv")

# Basic statistics
print("=== Dataset Statistics ===")
print(f"Total complaints: {len(df)}")
print(f"Unique categories: {df['category'].nunique()}")
print(f"Regions covered: {df['region'].nunique()}")
print(f"Duplicate complaints: {df['duplicate'].sum()}")

# Language comparison
def compare_languages(row):
    kz_len = len(row['text_kz']) if pd.notna(row['text_kz']) else 0
    ru_len = len(row['text_ru']) if pd.notna(row['text_ru']) else 0
    return kz_len, ru_len

df[['kz_length', 'ru_length']] = df.apply(lambda x: pd.Series(compare_languages(x)), axis=1)

print(f"\nAverage text length:")
print(f"Kazakh: {df['kz_length'].mean():.1f} characters")
print(f"Russian: {df['ru_length'].mean():.1f} characters")
```

### 📊 Data Visualization

```python
import plotly.express as px
import plotly.graph_objects as go

# Category distribution
fig1 = px.bar(df['category'].value_counts().reset_index(), 
              x='index', y='category',
              title="Complaints by Category",
              labels={'index': 'Category', 'category': 'Count'})
fig1.show()

# Urgency level distribution
fig2 = px.pie(df, names='urgency', 
              title="Distribution of Complaint Urgency Levels")
fig2.show()

# Regional distribution
fig3 = px.bar(df['region'].value_counts().reset_index(),
              x='index', y='region',
              title="Complaints by Region",
              labels={'index': 'Region', 'region': 'Count'})
fig3.update_xaxis(tickangle=45)
fig3.show()

# Status tracking
fig4 = px.sunburst(df, path=['category', 'status'], 
                   title="Complaint Status by Category")
fig4.show()
```

### 🔍 Bilingual Analysis

```python
# Analyze text length differences between languages
def analyze_bilingual_data(df):
    # Text length comparison
    df['kz_words'] = df['text_kz'].str.split().str.len()
    df['ru_words'] = df['text_ru'].str.split().str.len()
    
    print("=== Bilingual Analysis ===")
    print(f"Average words in Kazakh: {df['kz_words'].mean():.1f}")
    print(f"Average words in Russian: {df['ru_words'].mean():.1f}")
    
    # Length correlation
    correlation = df['kz_words'].corr(df['ru_words'])
    print(f"Length correlation between languages: {correlation:.3f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.scatter(df['kz_words'], df['ru_words'], alpha=0.6)
    plt.xlabel('Kazakh text length (words)')
    plt.ylabel('Russian text length (words)')
    plt.title('Text Length Comparison: Kazakh vs Russian')
    plt.plot([0, max(df['kz_words'].max(), df['ru_words'].max())], 
             [0, max(df['kz_words'].max(), df['ru_words'].max())], 
             'r--', alpha=0.5, label='Equal length line')
    plt.legend()
    plt.show()
    
    return df

df = analyze_bilingual_data(df)
```

### 🏛️ Government Response Analysis

```python
# Analyze government responses
def analyze_responses(df):
    # Response length analysis
    df['response_length'] = df['reply_text'].str.len()
    df['response_words'] = df['reply_text'].str.split().str.len()
    
    print("=== Government Response Analysis ===")
    print(f"Average response length: {df['response_length'].mean():.1f} characters")
    print(f"Average response words: {df['response_words'].mean():.1f} words")
    
    # Response time patterns (if timestamps were available)
    by_category = df.groupby('category')['response_words'].agg(['mean', 'std'])
    print("\nResponse length by category:")
    print(by_category.round(1))
    
    # Response quality indicators
    df['response_has_action'] = df['reply_text'].str.contains(
        'передана|ведется|решается|направлена|выполняется', case=False, na=False
    )
    
    action_rate = df['response_has_action'].mean()
    print(f"\nResponses indicating concrete action: {action_rate:.1%}")
    
    return df

df = analyze_responses(df)
```

---

## 📈 Applications

### 🎯 Use Cases

<div align="center">

| Application | Description | Complexity | Business Value |
|-------------|-------------|------------|----------------|
| 🤖 **Chatbot Training** | Train AI for government websites | Medium | High |
| 🌐 **Translation Systems** | Kazakh-Russian translation | Medium | High |
| 📊 **Sentiment Analysis** | Analyze citizen satisfaction | Low | Medium |
| 🔍 **Topic Classification** | Auto-categorize complaints | Low | High |
| 📈 **Trend Analysis** | Identify common issues | Low | High |
| 📝 **Response Templates** | Generate response templates | Medium | Medium |

</div>

### 🛠️ Implementation Examples

<details>
<summary><strong>Complaint Classification System</strong></summary>

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

def train_classifier(df):
    # Prepare data - combine Kazakh and Russian text
    df['combined_text'] = df['text_kz'] + ' ' + df['text_ru']
    
    # Split data
    X = df['combined_text']
    y = df['category']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create pipeline
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    classifier = MultinomialNB()
    
    # Train
    X_train_tfidf = vectorizer.fit_transform(X_train)
    classifier.fit(X_train_tfidf, y_train)
    
    # Test
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_tfidf)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump((vectorizer, classifier), 'complaint_classifier.pkl')
    
    return vectorizer, classifier

# Train the classifier
vectorizer, classifier = train_classifier(df)

# Use the classifier
def classify_complaint(text_kz, text_ru):
    combined = text_kz + ' ' + text_ru
    text_tfidf = vectorizer.transform([combined])
    prediction = classifier.predict(text_tfidf)[0]
    probability = classifier.predict_proba(text_tfidf).max()
    return prediction, probability

# Example usage
category, confidence = classify_complaint(
    "Мектепте интернет байланысы жоқ",
    "В школе нет интернета"
)
print(f"Predicted category: {category} (confidence: {confidence:.3f})")
```

</details>

<details>
<summary><strong>Urgency Detection</strong></summary>

```python
import re
from sklearn.ensemble import RandomForestClassifier

def extract_urgency_features(text_kz, text_ru):
    """Extract features that indicate urgency"""
    features = {}
    
    # Urgency keywords in Kazakh
    kz_urgent_words = ['тез', 'дереу', 'шұғыл', 'қауіпті', 'жедел']
    # Urgency keywords in Russian  
    ru_urgent_words = ['срочно', 'быстро', 'немедленно', 'опасно', 'критично']
    
    # Count urgent words
    features['kz_urgent_count'] = sum(1 for word in kz_urgent_words if word in text_kz.lower())
    features['ru_urgent_count'] = sum(1 for word in ru_urgent_words if word in text_ru.lower())
    
    # Text length (longer complaints might be more urgent)
    features['total_length'] = len(text_kz) + len(text_ru)
    
    # Exclamation marks
    features['exclamations'] = text_kz.count('!') + text_ru.count('!')
    
    # Capital letters ratio
    kz_caps = sum(1 for c in text_kz if c.isupper()) / max(len(text_kz), 1)
    ru_caps = sum(1 for c in text_ru if c.isupper()) / max(len(text_ru), 1)
    features['caps_ratio'] = (kz_caps + ru_caps) / 2
    
    return features

def train_urgency_classifier(df):
    # Extract features
    features_list = []
    for _, row in df.iterrows():
        features = extract_urgency_features(row['text_kz'], row['text_ru'])
        features_list.append(features)
    
    X = pd.DataFrame(features_list)
    y = df['urgency']
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    return clf

# Train urgency classifier
urgency_clf = train_urgency_classifier(df)

# Predict urgency for new complaint
def predict_urgency(text_kz, text_ru):
    features = extract_urgency_features(text_kz, text_ru)
    features_df = pd.DataFrame([features])
    prediction = urgency_clf.predict(features_df)[0]
    probability = urgency_clf.predict_proba(features_df).max()
    return prediction, probability

# Example
urgency, conf = predict_urgency(
    "Дереу көмек керек! Су жоқ!",
    "Срочно нужна помощь! Нет воды!"
)
print(f"Predicted urgency: {urgency} (confidence: {conf:.3f})")
```

</details>

---

## 📚 Examples

### 🎭 Real Data Samples

<details>
<summary><strong>Sample Record 1: Utilities Issue</strong></summary>

```json
{
  "id": 1,
  "text_kz": "Құрметті әкімдік, біздің ауылда жарық жоқ.",
  "text_ru": "Уважаемая акимат, в нашем селе нет света.",
  "category": "ЖКХ",
  "urgency": "высокая",
  "region": "Туркестанская область",
  "status": "новая",
  "reply_text": "Проблема передана в отдел энергетики, ведется работа.",
  "duplicate": false
}
```

**Analysis:**
- **Category**: ЖКХ (Utilities) - electricity outage
- **Urgency**: высокая (High) - no electricity is critical
- **Languages**: Perfect translation pair
- **Response**: Professional, indicates action taken

</details>

<details>
<summary><strong>Sample Record 2: Education Issue</strong></summary>

```json
{
  "id": 2,
  "text_kz": "Мектепте интернет байланысы жұмыс істемейді, балалар онлайн сабақ ала алмайды.",
  "text_ru": "В школе не работает интернет, дети не могут посещать онлайн уроки.",
  "category": "Образование",
  "urgency": "средняя",
  "region": "Алматинская область",
  "status": "в обработке",
  "reply_text": "Вопрос рассматривается совместно с отделом образования и техническими службами.",
  "duplicate": false
}
```

</details>

<details>
<summary><strong>Sample Record 3: Healthcare Issue</strong></summary>

```json
{
  "id": 3,
  "text_kz": "Ауырханада дәрігер жетіспейді, кезекте ұзақ уақыт күтеміз.",
  "text_ru": "В больнице не хватает врачей, долго ждем в очереди.",
  "category": "Здравоохранение",
  "urgency": "высокая",
  "region": "Жамбылская область",
  "status": "решена",
  "reply_text": "Проведено дополнительное укомплектование медицинского персонала.",
  "duplicate": false
}
```

</details>

### 📊 Data Quality Analysis

```python
def analyze_data_quality(df):
    """Comprehensive data quality analysis"""
    
    print("=== Data Quality Report ===")
    
    # Missing values
    missing_data = df.isnull().sum()
    if missing_data.any():
        print("Missing values:")
        for col, count in missing_data.items():
            if count > 0:
                print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    else:
        print("✅ No missing values found")
    
    # Duplicate analysis
    duplicates = df['duplicate'].sum()
    print(f"\nDuplicate complaints: {duplicates} ({duplicates/len(df)*100:.1f}%)")
    
    # Language consistency
    empty_kz = df['text_kz'].str.strip().eq('').sum()
    empty_ru = df['text_ru'].str.strip().eq('').sum()
    print(f"\nEmpty Kazakh texts: {empty_kz}")
    print(f"Empty Russian texts: {empty_ru}")
    
    # Category distribution
    print(f"\nCategory distribution:")
    for cat, count in df['category'].value_counts().items():
        print(f"  {cat}: {count} ({count/len(df)*100:.1f}%)")
    
    # Urgency distribution
    print(f"\nUrgency distribution:")
    for urg, count in df['urgency'].value_counts().items():
        print(f"  {urg}: {count} ({count/len(df)*100:.1f}%)")
    
    # Status distribution
    print(f"\nStatus distribution:")
    for status, count in df['status'].value_counts().items():
        print(f"  {status}: {count} ({count/len(df)*100:.1f}%)")
    
    return df

# Run quality analysis
df = analyze_data_quality(df)
```

---

## ⚠️ Limitations

### 🚨 Data Limitations

<div align="center">

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Small Dataset** | Limited training data (256KB) | Combine with other datasets |
| **Regional Bias** | May not represent all regions equally | Collect more regional data |
| **Temporal Scope** | Snapshot from specific time period | Regular updates needed |
| **Category Coverage** | Limited government service categories | Expand category coverage |
| **Response Quality** | Varies by government department | Quality control measures |

</div>

### 🔍 Technical Considerations

<details>
<summary><strong>Data Quality Issues</strong></summary>

**Language Quality:**
- Translation quality may vary between records
- Some Kazakh text may contain Russian loanwords
- Formal government language style in responses
- Regional dialect variations not captured

**Representativeness:**
- Small sample size limits statistical power
- May not represent all types of citizen complaints
- Government response quality varies by department
- Temporal patterns not captured in static dataset

**Technical Limitations:**
- Limited metadata for advanced analysis
- No timestamp information for trend analysis
- No demographic information about complainants
- Response effectiveness not tracked

</details>

<details>
<summary><strong>Ethical Considerations</strong></summary>

**Privacy:**
- Personal information should be removed/anonymized
- Geographic information limited to oblast level
- No personally identifiable information included
- Complaint content should be reviewed for sensitivity

**Bias and Fairness:**
- May reflect biases in government response patterns
- Urban vs rural representation may be uneven
- Language preference patterns may not be representative
- Response quality may vary by complaint type

**Usage Guidelines:**
- Validate models with domain experts before deployment
- Consider cultural context when analyzing text
- Ensure compliance with local data protection laws
- Use for research and development purposes primarily

</details>

---

## 🤝 Contributing

### 🛠️ How to Contribute

<div align="center">

| Contribution Type | Description | Skills Required |
|------------------|-------------|-----------------|
| 🐛 **Data Quality** | Report data errors, inconsistencies | Basic |
| 🌐 **Translation** | Improve Kazakh-Russian translations | Native speaker |
| 📊 **Analysis** | Add data analysis scripts | Python, statistics |
| 🔧 **Tools** | Create processing utilities | Programming |
| 📝 **Documentation** | Improve guides and examples | Writing |
| 📈 **Expansion** | Add more complaint categories | Domain expertise |

</div>

### 📋 Contributing Process

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b improve-translations`)
3. **Make** your improvements
4. **Test** your changes thoroughly
5. **Commit** with clear messages (`git commit -m 'Improve Kazakh translations'`)
6. **Push** to your branch (`git push origin improve-translations`)
7. **Submit** a Pull Request

### 🎯 Priority Areas

- **Data Quality**: Review and improve translations
- **Category Expansion**: Add more government service areas
- **Regional Coverage**: Include more regions of Kazakhstan
- **Response Analysis**: Improve government response classification
- **Documentation**: Add more usage examples and tutorials

---

## 📞 Support & Contact

<div align="center">

### 💬 Get Help

| Platform | Purpose | Response Time |
|----------|---------|---------------|
| **GitHub Issues** | Bug reports, data issues | 1-2 days |
| **GitHub Discussions** | Questions, suggestions | 2-3 days |
| **Email** | Private inquiries | 3-5 days |

### 👤 Maintainer

**Adilzhan Baidalin** ([@AdilzhanB](https://github.com/AdilzhanB))

- **GitHub**: [@AdilzhanB](https://github.com/AdilzhanB)
- **Created**: June 18, 2025
- **Dataset Size**: 256 KB

</div>

---

## 📜 License

<div align="center">

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

### Usage Rights

- ✅ **Commercial use** - Use in commercial projects
- ✅ **Modification** - Modify and improve the dataset
- ✅ **Distribution** - Share and redistribute
- ✅ **Private use** - Use for private projects
- ❌ **Liability** - No warranty provided
- ❌ **Warranty** - Use at your own risk

</div>

---

## 🙏 Acknowledgments

<div align="center">

Special thanks to:

- **Government of Kazakhstan** for public service inspiration
- **Citizens of Kazakhstan** who interact with government services
- **Open source community** for tools and libraries
- **Linguistic experts** for translation guidance
- **GitHub community** for hosting and collaboration tools

</div>

---

<div align="center">

### 📊 Quick Stats

![GitHub repo size](https://img.shields.io/github/repo-size/AdilzhanB/KZgovData?style=flat-square&color=blue)
![Dataset size](https://img.shields.io/badge/Dataset%20Size-256KB-orange?style=flat-square)
![Languages](https://img.shields.io/badge/Languages-2-green?style=flat-square)
![Records](https://img.shields.io/badge/Records-Variable-purple?style=flat-square)

---

### 🚀 Ready to analyze Kazakhstan government data?

**[Download Dataset](#-download)** • **[View Examples](#-examples)** • **[Start Contributing](#-contributing)**

---

*Building bridges between citizens and government through data science*

*Last updated: June 18, 2025 • Dataset size: 256KB • Created by [@AdilzhanB](https://github.com/AdilzhanB)*

</div>
