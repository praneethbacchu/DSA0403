import pandas as pd
import matplotlib.pyplot as plt

# 1. Create dataset and store in a CSV file
csv_filename = "soccer_players.csv"
players_df = pd.DataFrame({
    "name": [
        "Liam Vance", "Mateo Silva", "Noah Sterling", "Lucas Meyer",
        "Ethan Brooks", "Julian Rossi", "Oliver Grant", "Daniel Kim",
        "Samuel Adeyemi", "Hugo Laurent"
    ],
    "age": [24, 29, 22, 31, 27, 26, 33, 21, 28, 25],
    "position": [
        "Forward", "Midfielder", "Forward", "Defender",
        "Midfielder", "Forward", "Defender", "Goalkeeper",
        "Forward", "Midfielder"
    ],
    "goals_scored": [22, 11, 18, 4, 14, 19, 3, 0, 16, 9],
    "weekly_salary": [140000, 115000, 95000, 105000, 120000, 130000, 98000, 65000, 125000, 88000]
})
players_df.to_csv(csv_filename, index=False)

# 2. Read CSV into pandas DataFrame
df = pd.read_csv(csv_filename)

# 3. Top 5 by goals and top 5 by salary
top_5_goals = df.nlargest(5, "goals_scored")[["name", "position", "goals_scored"]]
top_5_salary = df.nlargest(5, "weekly_salary")[["name", "position", "weekly_salary"]]

print("Top 5 Players by Goals Scored:")
print(top_5_goals.to_string(index=False))

print("\nTop 5 Players by Weekly Salary:")
print(top_5_salary.to_string(index=False))

# 4. Average age and players above average age
avg_age = df["age"].mean()
above_avg_players = df[df["age"] > avg_age]["name"].tolist()

print(f"\nAverage Age: {avg_age:.1f} years")
print("Players Above Average Age:", ", ".join(above_avg_players))

# 5. Bar chart of players by position
pos_counts = df["position"].value_counts()
plt.figure(figsize=(7, 4))
pos_counts.plot(kind="bar", color="mediumseagreen")
plt.title("Player Distribution by Position")
plt.xlabel("Position")
plt.ylabel("Number of Players")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
