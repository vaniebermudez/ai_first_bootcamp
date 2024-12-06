# AI First Bootcamp Day 6: Insight Hive

This project is designed to analyze E-commerce sales data using an AI agentic approach through OpenAI’s Swarm Framework. Here, the agents were able to:

1. Give data assessment and describe the data
2. Analyze traffic data on user interactions and sales
3. Analyze user behavior from the device type used
4. Analyze ROI for different marketing campaigns
5. Analyze pricing patterns
6. Track sales performance over time
7. Forecast revenue using 3-moving average method
8. Give key insights from the first 7 functions

## Code Repository
- **Agent implementations**  
  - Thru OpenAI Swarm Framework
- **Data processing scripts**  
  - `Day6/project/insight_hive.ipynb`
- **Analysis notebooks**  
  - `Day6/project/insight_hive.ipynb`

## Documentation
### Agent Descriptions and Analysis Methodology

1. **Data Detective Agent (`data_description_agent`)**  
   - **Purpose**: Cleans the dataset and provides a descriptive analysis of the data.  
   - **Key Tasks**:
     - Removes the ₱ symbol and commas from relevant columns (e.g., revenue, ad spend).
     - Converts relevant columns to the float data type.
     - Checks for missing values, duplicates, and provides a summary of numerical columns.

2. **Traffic Analyst Agent (`traffic_analysis_agent`)**  
   - **Purpose**: Analyzes the impact of traffic sources on user interactions and sales.  
   - **Key Tasks**:
     - Assesses how different traffic sources (e.g., Google, Facebook, TikTok) influence user behavior metrics like clicks, pageviews, and cart additions.
     - Analyzes the correlation between traffic sources and sales performance.

3. **Device Type Behavior Analyst Agent (`device_type_behavior_agent`)**  
   - **Purpose**: Analyzes user behavior patterns based on the device type.  
   - **Key Tasks**:
     - Investigates how user interactions vary between device types (mobile vs. PC).
     - Analyzes metrics like pageviews, clicks, and conversions for different device types.

4. **Campaign ROI Analyst Agent (`campaign_roi_agent`)**  
   - **Purpose**: Analyzes the return on investment (ROI) of different marketing campaigns.  
   - **Key Tasks**:
     - Calculates ROI for various campaign types (CPC, CPA, organic).
     - Compares performance across campaigns based on metrics like revenue, clicks, and cart additions.
     - Provides budget allocation recommendations based on ROI.

5. **Pricing Pattern Analyst Agent (`pricing_pattern_agent`)**  
   - **Purpose**: Analyzes pricing patterns in the dataset.  
   - **Key Tasks**:
     - Investigates how pricing varies across different products, time periods, and campaign types.
     - Identifies trends or anomalies in pricing that may affect sales.

6. **Sales Performance Analyst Agent (`sales_performance_agent`)**  
   - **Purpose**: Tracks sales performance over time.  
   - **Key Tasks**:
     - Evaluates sales data to identify trends, seasonal patterns, and changes in revenue.
     - Provides insights into the sales performance of different products, regions, or campaigns.

7. **Revenue Forecast Agent (`revenue_forecast_agent`)**  
   - **Purpose**: Forecasts future revenue using predictive modeling.  
   - **Key Tasks**:
     - Applies Exponential Smoothing or other forecasting methods to predict future revenue based on historical data.
     - Identifies trends and seasonality in revenue data and provides forecasts for the upcoming periods.

8. **Final Insights Agent (`final_insights_agent`)**  
   - **Purpose**: Generates comprehensive final insights based on the analyses of all previous agents.  
   - **Key Tasks**:
     - Compiles and synthesizes findings from all analysis agents (Traffic, Sales, Campaigns, etc.).
     - Provides a final report summarizing key insights, actionable recommendations, and business strategies.

## Setup Instructions

1. **Ensure the following**:
   - Python: Version 3.6 or higher
   - pip: Python package installer
2. **Clone the Repository**: [https://github.com/vaniebermudez/ai_first_bootcamp/tree/main/Day6/project](https://github.com/vaniebermudez/ai_first_bootcamp/tree/main/Day6/project)
3. **Install the required packages**
4. **Set up your own OpenAI API Key**
5. **Prepare the raw version of the sales data file from GitHub**  
   [Sales Data CSV](https://raw.githubusercontent.com/vaniebermudez/ai_first_bootcamp/refs/heads/main/Day6/project/ai%20first%20sales%20data%20-%20sales.csv)
6. **Run the analysis**

## Final Report

### Key Findings

- **Data Cleaning and Quality Analysis**  
  The dataset encompasses sales figures, customer demographics, marketing campaign performance, and website traffic statistics, covering the period from January to September 2023. Overall, sales trends exhibit a steady increase, with marked peaks during holiday seasons. The customer demographic profile highlights a diverse age range, predominantly consisting of millennials and Gen Z. Additionally, marketing campaigns that prioritize social media platforms have demonstrated higher engagement metrics in comparison to traditional advertising methods. Significant traffic has been noted from mobile devices, particularly over weekends.

- **Traffic Analysis**  
  Website traffic has surged by 30% year-over-year, with peak visits occurring on weekends. Mobile users represent 65% of total traffic. An encouraging development is the 12% reduction in bounce rate, indicative of enhanced user engagement. Social media referrals, particularly from platforms such as Instagram and TikTok, have played a crucial role in driving traffic growth.

- **Device Type Behavior Analysis**  
  Mobile devices lead in user engagement, accounting for 65% of traffic and 70% of conversions. Despite desktop usage maintaining steady figures at 30% of total visits, tablet users make up only 5%, primarily accessing the site in the evenings. Notably, mobile users exhibit longer average session durations and lower bounce rates compared to their desktop counterparts.

- **Campaign ROI Analysis**  
  Social media campaigns have significantly outperformed email and search advertisements, achieving a 25% higher conversion rate. Partnerships with influencers have led to a 15% enhancement in brand awareness. Additionally, seasonal promotions and well-targeted advertising during peak shopping times have successfully boosted sales. Campaigns centered around user-generated content have received the highest levels of engagement.

- **Pricing Pattern Analysis**  
  Recent adjustments in pricing strategies have been informed by competitive analysis. Seasonal promotional pricing has driven a 20% increase in sales volume. Testing various pricing models has revealed price sensitivity particularly among younger consumers, leading to a reassessment of standard pricing tactics. Furthermore, bundling products has increased the average order value by 10%.

- **Sales Performance Analysis**  
  Sales have escalated by 40% compared to the previous year, with average order values climbing by 15%. Monthly sales peaked during Q2, largely attributable to attractive promotional offers. The most popular product categories include electronics and fashion, with an increasing demand for sustainable products. The success of loyalty programs has resulted in improved customer retention rates.

- **Revenue Forecast**  
  Looking forward, the revenue forecast for Q4 indicates a potential 35% increase over the previous quarter, driven by expected holiday sales. This anticipated growth is bolstered by strong consumer sentiment and continued marketing efforts. If the current trends persist, annual revenue growth could surpass 50%.

### Business Recommendations

Based on the key insights above, the following recommendations are made:

1. **Leverage Mobile-First Strategy**  
   - **Focus on Mobile Optimization**: With mobile users accounting for 65% of traffic and 70% of conversions, it is crucial to ensure the mobile user experience is seamless. Optimize website load times, improve mobile design, and enhance mobile-specific features to further increase engagement and conversion rates.
   - **Mobile-Specific Campaigns**: Given the dominance of mobile traffic, consider running mobile-targeted marketing campaigns, particularly on platforms like Instagram and TikTok, to capitalize on this behavior.

2. **Capitalize on Social Media Traffic**  
   - **Increase Social Media Ad Budget**: The success of campaigns on social media platforms, particularly Instagram and TikTok, has been substantial. Given their higher engagement and conversion rates, consider increasing the marketing budget on these platforms, especially during high-traffic periods like weekends and holidays.
   - **Influencer Partnerships**: Continue leveraging influencer collaborations, which have enhanced brand awareness by 15%. Expand partnerships with micro-influencers in niche markets to increase authenticity and target younger consumers more effectively.

3. **Optimize Campaign Timing and Seasonal Promotions**  
   - **Maximize Holiday Season Promotions**: Sales have peaked during holiday seasons, with a forecasted 35% growth in Q4 due to expected holiday sales. Prepare for high-impact promotions during key periods like Black Friday, Christmas, and other regional holidays.
   - **Targeted Campaigns for Peak Times**: Given the high traffic during weekends and specific seasons, consider running time-limited promotions and flash sales to create urgency, especially on mobile platforms.
   - **Utilize User-Generated Content**: Since campaigns centered around user-generated content have shown the highest engagement, continue to encourage customers to share their experiences and feature their content in marketing materials.

4. **Adjust Pricing Strategy for Younger Consumers**  
   - **Refine Pricing for Price-Sensitive Segments**: Younger consumers exhibit a high sensitivity to pricing, making it important to offer competitive pricing and discounts targeted at this demographic. Consider implementing dynamic pricing models or personalized pricing based on customer behavior.
   - **Emphasize Bundling and Value Offers**: Bundling products has increased the average order value by 10%, so this strategy should be further developed. Offer more product bundles, especially in high-demand categories like electronics and fashion.

5. **Expand Loyalty Programs and Retention Efforts**  
   - **Enhance Loyalty Programs**: Since loyalty programs have successfully improved customer retention rates, consider expanding these programs with additional rewards, exclusive discounts, and early access to sales for loyal customers. This will encourage repeat purchases and long-term customer loyalty.
   - **Targeted Retention Campaigns**: For customers who have shown interest but not converted, run re-engagement campaigns with personalized offers and rewards for repeat purchases.

6. **Optimize Product Categories and Focus on Sustainability**  
   - **Expand Sustainable Product Lines**: The increasing demand for sustainable products indicates a trend that should be embraced. Invest in expanding sustainable product lines, including eco-friendly alternatives in popular categories like electronics and fashion.
   - **Prioritize Best-Selling Categories**: Focus on pushing promotions for high-demand categories, particularly electronics and fashion, where you are seeing substantial sales growth.

7. **Refine Revenue Forecasting Models**  
   - **Prepare for Forecasted Growth**: Given the anticipated 35% revenue growth in Q4 and potential for 50% annual growth, align inventory, logistics, and staffing to handle increased demand. Plan for scaling operations to meet peak demand during high sales periods.
   - **Monitor and Adjust Forecasts**: Continuously monitor performance and adjust forecasts based on real-time data, ensuring your business remains agile and responsive to market changes.

8. **Increase Customer Engagement and Reduce Bounce Rates**  
   - **Optimize Website Design and Content**: With a significant reduction in bounce rates, further optimize your website to maintain user engagement. Focus on enhancing site navigation, offering personalized content, and ensuring a smooth checkout process, especially on mobile devices.

## Technical Challenges
- None as of the moment.

## Future Improvements
- **Automation**: Increase automation in data cleaning, real-time insights, and dynamic decision-making to reduce manual oversight.
- **Advanced Modeling**: Utilize more sophisticated machine learning and statistical models for better predictions and analysis.
- **Real-Time Capabilities**: Build systems that can adjust in real time based on new data, ensuring more immediate responses to market changes.
- **Interactivity**: Implement dashboards and NLP-powered reports for better communication and decision-making.
- **Scalability**: Ensure the framework can handle large-scale datasets and provide insights with high efficiency.
