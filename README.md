# Collaborative-Based Recommendation System

## 📌 Description
This project implements a **collaborative filtering recommendation system** that suggests items to users based on **user-to-user similarity** and **item-to-item similiarity**.. It uses **pivot table** and similarity metrics to provide recommendations without requiring content details. This is a **backend system**, no UI included.

---

## 🚀 Features
- User-based and item-based collaborative filtering
- Use pivot table
- Cosine similarity for recommendations
- Works for large datasets

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- scikit-learn (similarity metrics)
- Surprise 
- Jupyter Notebook

---

## 📂 Project Structure
collaborative-recommender/
│── data/
│── notebooks/
│   └── collaborative recommendation system.ipynb
│── model/
│   ├── 📂 collaborative_system        
│   └── all_movies.pkl
│   └── movie_data.pkl
│   └── pivot_item.pkl
│   └── pivot_user.pkl
│   └── item_based.pkl
│   └── user_based.pkl
│   └── popular_movie.pkl
│   └── search_text_database.pkl
│   └── genre_database.pkl
│                              
│── requirements.txt
│── README.md

---

## ⚙️ Installation
```bash
git clone https://github.com/YOUR-USERNAME/collaborative-recommender.git
```



## 📊 Model Details
* Pivot table : for user-to-user / item-to-item
* user similarity : cosine similarity for user based recommendations
* item similarity : cosine similarity for item based recommendations
* SBERT embedding : for better recommendations
* Faiss : Store embedding in faiss
---

