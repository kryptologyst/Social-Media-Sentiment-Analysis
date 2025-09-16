"""
Mock Social Media Data Generator
Generates diverse social media posts for sentiment analysis training
"""

import pandas as pd
import random

def generate_mock_data():
    """Generate a comprehensive dataset of social media posts with sentiments"""
    
    # Positive posts
    positive_posts = [
        "I absolutely love this new feature! It's amazing! 🎉",
        "Best app ever! So smooth and user-friendly!",
        "Fantastic update! Everything works perfectly now!",
        "Great job on the new design! Looks incredible!",
        "This app has changed my life for the better!",
        "Outstanding customer service! They fixed my issue instantly!",
        "Love the new interface! So intuitive and clean!",
        "Perfect! Exactly what I was looking for!",
        "Amazing performance improvements! Much faster now!",
        "Brilliant work by the development team!",
        "This is exactly what I needed! Thank you!",
        "Incredible features! Worth every penny!",
        "So happy with this purchase! Highly recommend!",
        "Excellent quality and great value!",
        "This app rocks! Using it every day now!",
        "Superb functionality! Everything I wanted!",
        "Love how easy it is to use!",
        "Great experience! Will definitely use again!",
        "Awesome job on the latest update!",
        "Perfect solution to my problem!",
        "Really impressed with the quality!",
        "This is fantastic! Exceeded my expectations!",
        "Love the new features! So helpful!",
        "Great work! Keep it up!",
        "This app is a game changer!",
        "Wonderful experience! Highly satisfied!",
        "Excellent performance! No complaints!",
        "Love the simplicity and effectiveness!",
        "This is exactly what I was hoping for!",
        "Amazing results! Very pleased!"
    ]
    
    # Negative posts
    negative_posts = [
        "Worst update ever! Everything is broken now!",
        "This app is terrible! Crashes constantly!",
        "Completely disappointed with this purchase!",
        "Horrible user experience! So frustrating!",
        "This is useless! Waste of time and money!",
        "Terrible customer service! No help at all!",
        "Buggy mess! Nothing works properly!",
        "Awful design! So confusing to navigate!",
        "This app is a disaster! Avoid at all costs!",
        "Extremely disappointed! Not what was promised!",
        "Terrible performance! So slow and laggy!",
        "This is broken! Doesn't work as advertised!",
        "Horrible experience! Will never use again!",
        "Complete waste of money! Total ripoff!",
        "This app is garbage! Uninstalling immediately!",
        "Terrible quality! Very poor workmanship!",
        "Awful support! They don't care about users!",
        "This is frustrating! Nothing works right!",
        "Horrible interface! So hard to use!",
        "Terrible update! Ruined everything!",
        "This is annoying! Constant problems!",
        "Awful experience! Very dissatisfied!",
        "This app sucks! So many issues!",
        "Terrible functionality! Doesn't do what it says!",
        "Horrible bugs! Crashes all the time!",
        "This is pathetic! Expected much better!",
        "Awful performance! Extremely slow!",
        "Terrible design choices! Makes no sense!",
        "This is broken! Needs major fixes!",
        "Horrible user interface! Very confusing!"
    ]
    
    # Neutral posts
    neutral_posts = [
        "The app works fine. Nothing special though.",
        "It's okay, does what it's supposed to do.",
        "Average performance. Could be better, could be worse.",
        "The update is alright. Some improvements, some issues.",
        "It's functional but not particularly impressive.",
        "Works as expected. No major complaints or praise.",
        "The interface is standard. Nothing groundbreaking.",
        "It does the job. Not amazing, not terrible.",
        "Decent app overall. Room for improvement.",
        "It's fine for basic use. Nothing more, nothing less.",
        "The features are adequate for my needs.",
        "Standard functionality. What you'd expect.",
        "It works but there are better alternatives.",
        "Acceptable quality. Gets the job done.",
        "The app is functional but uninspiring.",
        "It's okay for occasional use.",
        "Average user experience. Nothing stands out.",
        "The performance is acceptable but not great.",
        "It does what it says. No surprises.",
        "Mediocre app. Neither good nor bad.",
        "The design is plain but functional.",
        "It works fine for basic tasks.",
        "Standard features. Nothing innovative.",
        "The app is usable but not exciting.",
        "It's an okay solution for the price.",
        "Functional but lacks polish.",
        "The interface is simple and straightforward.",
        "It works but feels outdated.",
        "Adequate for basic needs.",
        "The app is stable but unremarkable."
    ]
    
    # Create balanced dataset
    posts = []
    sentiments = []
    
    # Add all posts with their sentiments
    for post in positive_posts:
        posts.append(post)
        sentiments.append('Positive')
    
    for post in negative_posts:
        posts.append(post)
        sentiments.append('Negative')
    
    for post in neutral_posts:
        posts.append(post)
        sentiments.append('Neutral')
    
    # Create DataFrame
    df = pd.DataFrame({
        'Post': posts,
        'Sentiment': sentiments
    })
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def save_mock_data(filename='social_media_posts.csv'):
    """Save the mock data to a CSV file"""
    df = generate_mock_data()
    df.to_csv(filename, index=False)
    print(f"Mock data saved to {filename}")
    print(f"Dataset contains {len(df)} posts:")
    print(df['Sentiment'].value_counts())
    return df

if __name__ == "__main__":
    # Generate and save the data
    data = save_mock_data()
    print("\nFirst 5 rows:")
    print(data.head())
