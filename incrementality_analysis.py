import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

# Note: In a real environment, we would use the openai or google-generativeai SDK.
# For this PoW, I simulated the LLM API call to show the architecture.

def generate_synthetic_data(n_users=5000):
    """
    Generates synthetic user-level streaming data for a Difference-in-Differences analysis.
    """
    np.random.seed(42)
    
    # Create user base
    users = pd.DataFrame({
        'user_id': range(1, n_users + 1),
        # 50% treatment (get video), 50% control (audio only)
        'is_treatment': np.random.choice([0, 1], size=n_users)
    })
    
    # Pre-period (Audio only for everyone)
    pre_period = users.copy()
    pre_period['is_post'] = 0
    # Base streaming minutes (gamma distribution for realistic skew)
    pre_period['total_minutes'] = np.random.gamma(shape=2.0, scale=30.0, size=n_users)
    
    # Post-period
    post_period = users.copy()
    post_period['is_post'] = 1
    
    # Calculate post-period minutes with treatment effect
    # Base trend: Everyone streams 5 minutes more in the post period
    # Treatment effect: Video access adds a true incremental 15 minutes of streaming
    base_trend = 5.0
    true_treatment_effect = 15.0
    
    post_period['total_minutes'] = (
        np.random.gamma(shape=2.0, scale=30.0, size=n_users) + 
        base_trend + 
        (post_period['is_treatment'] * true_treatment_effect)
    )
    
    # Combine into panel dataset
    df = pd.concat([pre_period, post_period], ignore_index=True)
    
    # Create the interaction term for the DiD regression
    df['treatment_x_post'] = df['is_treatment'] * df['is_post']
    
    return df

def run_causal_inference(df):
    """
    Runs a Difference-in-Differences OLS regression.
    """
    # Y = B0 + B1*Treatment + B2*Post + B3*(Treatment*Post)
    model = smf.ols('total_minutes ~ is_treatment + is_post + treatment_x_post', data=df)
    results = model.fit()
    
    # Extract the causal impact metrics
    causal_lift = results.params['treatment_x_post']
    p_value = results.pvalues['treatment_x_post']
    ci_lower = results.conf_int().loc['treatment_x_post'][0]
    ci_upper = results.conf_int().loc['treatment_x_post'][1]
    
    stats_dict = {
        "incremental_minutes": round(causal_lift, 2),
        "p_value": round(p_value, 4),
        "confidence_interval": f"[{round(ci_lower, 2)}, {round(ci_upper, 2)}]",
        "is_significant": p_value < 0.05
    }
    
    return stats_dict, results.summary()

def generate_stakeholder_summary_via_llm(stats_dict):
    """
    Simulates sending the statistical output to an LLM to generate dashboard text 
    for editorial and content strategy teams.
    """
    
    prompt = f"""
    You are a data science assistant for the Spotify Podcast & Video Analytics team.
    Translate these causal inference results into a 3-sentence summary for a non-technical 
    Editorial Lead. Focus on the business impact. Do not use terms like "p-value" or "Difference-in-Differences".

    Data to translate:
    - Incremental minutes driven by video: {stats_dict['incremental_minutes']}
    - Statistically significant: {stats_dict['is_significant']}
    - Confidence Interval: {stats_dict['confidence_interval']}
    """
    
    # Simulated LLM response for the sake of the GitHub repo demonstration
    llm_response = f"""
    *** GenAI Dashboard Summary Draft ***
    Adding video to the podcast was highly successful, driving a true incremental lift of {stats_dict['incremental_minutes']} minutes of total streaming time per user. We are confident that this increase is directly caused by the video feature rather than natural baseline growth. Given these strong results, the content strategy team should feel confident expanding the video format to similar shows in the portfolio.
    """
    
    return llm_response

if __name__ == "__main__":
    print("1. Generating synthetic streaming data...")
    df = generate_synthetic_data()
    
    print("2. Running Difference-in-Differences Causal Model...")
    stats, summary = run_causal_inference(df)
    
    print("\n--- STATISTICAL OUTPUT ---")
    print(f"Estimated Causal Lift: +{stats['incremental_minutes']} mins")
    print(f"P-Value: {stats['p_value']} (Significant: {stats['is_significant']})")
    
    print("\n3. Generating Stakeholder Summary via LLM...")
    dashboard_text = generate_stakeholder_summary_via_llm(stats)
    print(dashboard_text)
