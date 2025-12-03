# churn_analyzer.py - Advanced analytics for churn prediction results

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

class ChurnAnalyzer:
    """Class to perform advanced analytics on churn prediction results"""
    
    def __init__(self, results_df=None):
        """Initialize the analyzer
        
        Args:
            results_df (DataFrame, optional): DataFrame with prediction results
        """
        self.results_df = results_df
        self.analysis_results = {}
        self.output_dir = 'static/analysis'
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self, results_df):
        """Load prediction results data
        
        Args:
            results_df (DataFrame): DataFrame with prediction results
        """
        self.results_df = results_df
    
    def segment_customers(self, n_clusters=3):
        """Segment customers based on their features
        
        Args:
            n_clusters (int): Number of clusters to create
            
        Returns:
            dict: Segmentation results
        """
        if self.results_df is None:
            return {'error': 'No data loaded'}
        
        try:
            # Select numeric features for clustering
            numeric_cols = self.results_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Remove target and ID columns
            exclude_cols = ['customer_id', 'customerID', 'churn_probability']
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            # Ensure we have enough features
            if len(feature_cols) < 2:
                return {'error': 'Not enough numeric features for clustering'}
            
            # Normalize data
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.results_df[feature_cols])
            
            # Apply PCA for dimensionality reduction
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            
            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            
            # Add cluster labels to the dataframe
            self.results_df['customer_segment'] = clusters
            
            # Calculate cluster statistics
            cluster_stats = self.results_df.groupby('customer_segment').agg({
                'churn_probability': ['mean', 'count'],
                **{col: 'mean' for col in feature_cols}
            })
            
            # Flatten multi-index columns
            cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
            
            # Create segment names based on churn probability
            segment_mapping = {}
            churn_means = self.results_df.groupby('customer_segment')['churn_probability'].mean()
            
            # Sort clusters by churn probability
            sorted_clusters = churn_means.sort_values().index
            segment_names = ['Loyal Customers', 'At-Risk Customers', 'High-Risk Customers']
            
            for i, cluster in enumerate(sorted_clusters):
                if i < len(segment_names):
                    segment_mapping[cluster] = segment_names[i]
                else:
                    segment_mapping[cluster] = f'Segment {cluster+1}'
            
            # Map segment names to the dataframe
            self.results_df['segment_name'] = self.results_df['customer_segment'].map(segment_mapping)
            
            # Store results
            self.analysis_results['segmentation'] = {
                'cluster_stats': cluster_stats.to_dict(),
                'segment_mapping': segment_mapping,
                'pca_result': pca_result.tolist(),
                'feature_importance': dict(zip(feature_cols, pca.components_[0]))
            }
            
            return self.analysis_results['segmentation']
            
        except Exception as e:
            return {'error': f'Error in segmentation: {str(e)}'}
    
    def analyze_churn_factors(self):
        """Analyze the factors contributing to churn
        
        Returns:
            dict: Churn factors analysis results
        """
        if self.results_df is None:
            return {'error': 'No data loaded'}
        
        try:
            # Count occurrences of each reason
            all_reasons = []
            for reasons in self.results_df['top_reasons']:
                all_reasons.extend(reasons)
            
            reason_counts = {}
            for reason in all_reasons:
                if reason in reason_counts:
                    reason_counts[reason] += 1
                else:
                    reason_counts[reason] = 1
            
            # Sort by count
            sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Calculate percentage
            total_reasons = sum(reason_counts.values())
            reason_percentages = {reason: (count / total_reasons) * 100 for reason, count in reason_counts.items()}
            
            # Analyze reasons by segment
            segment_reasons = {}
            if 'segment_name' in self.results_df.columns:
                for segment in self.results_df['segment_name'].unique():
                    segment_df = self.results_df[self.results_df['segment_name'] == segment]
                    
                    segment_all_reasons = []
                    for reasons in segment_df['top_reasons']:
                        segment_all_reasons.extend(reasons)
                    
                    segment_reason_counts = {}
                    for reason in segment_all_reasons:
                        if reason in segment_reason_counts:
                            segment_reason_counts[reason] += 1
                        else:
                            segment_reason_counts[reason] = 1
                    
                    # Sort by count
                    segment_sorted_reasons = sorted(segment_reason_counts.items(), key=lambda x: x[1], reverse=True)
                    segment_reasons[segment] = segment_sorted_reasons
            
            # Store results
            self.analysis_results['churn_factors'] = {
                'reason_counts': reason_counts,
                'reason_percentages': reason_percentages,
                'sorted_reasons': sorted_reasons,
                'segment_reasons': segment_reasons
            }
            
            return self.analysis_results['churn_factors']
            
        except Exception as e:
            return {'error': f'Error in churn factors analysis: {str(e)}'}
    
    def generate_retention_strategies(self):
        """Generate personalized retention strategies based on segments and churn factors
        
        Returns:
            dict: Retention strategies
        """
        if self.results_df is None or 'segment_name' not in self.results_df.columns:
            return {'error': 'Segmentation must be performed first'}
        
        try:
            # Define base strategies for each segment
            base_strategies = {
                'Loyal Customers': [
                    'Loyalty rewards program',
                    'Early access to new features',
                    'Referral incentives'
                ],
                'At-Risk Customers': [
                    'Proactive customer service check-ins',
                    'Targeted discounts on current services',
                    'Service upgrades at reduced cost'
                ],
                'High-Risk Customers': [
                    'Immediate outreach by customer success team',
                    'Substantial contract renewal incentives',
                    'Personalized retention offers based on usage patterns'
                ]
            }
            
            # Customize strategies based on top churn reasons
            if 'churn_factors' in self.analysis_results:
                reason_strategies = {
                    'High monthly charges': [
                        'Offer 15% discount for 12-month commitment',
                        'Introduce tiered pricing plan',
                        'Bundle services for cost savings'
                    ],
                    'No long-term contract': [
                        'Highlight benefits of 12-month contracts',
                        'Offer one-time bonus for contract signup',
                        'Create flexible contract options'
                    ],
                    'Competitor offers': [
                        'Competitive price matching',
                        'Highlight unique value propositions',
                        'Offer exclusive features not available with competitors'
                    ],
                    'Service dissatisfaction': [
                        'Immediate service quality review',
                        'Dedicated support representative',
                        'Service credits for past issues'
                    ],
                    'Lack of features': [
                        'Free trial of premium features',
                        'Customized feature package',
                        'Early access to beta features'
                    ],
                    'Technical issues': [
                        'Priority technical support',
                        'System audit and optimization',
                        'Automatic monitoring and issue resolution'
                    ]
                }
                
                # Get top reasons for each segment
                segment_top_reasons = {}
                for segment, reasons in self.analysis_results['churn_factors']['segment_reasons'].items():
                    if reasons:  # Check if there are any reasons
                        segment_top_reasons[segment] = [reason for reason, _ in reasons[:2]]  # Top 2 reasons
            
            # Generate personalized strategies for each customer
            self.results_df['retention_strategies'] = self.results_df.apply(
                lambda row: self._get_customer_strategies(row, base_strategies, 
                                                         segment_top_reasons, reason_strategies),
                axis=1
            )
            
            # Summarize strategies by segment
            segment_strategies = {}
            for segment in self.results_df['segment_name'].unique():
                segment_df = self.results_df[self.results_df['segment_name'] == segment]
                
                # Get all strategies for this segment
                all_strategies = []
                for strategies in segment_df['retention_strategies']:
                    all_strategies.extend(strategies)
                
                # Count occurrences
                strategy_counts = {}
                for strategy in all_strategies:
                    if strategy in strategy_counts:
                        strategy_counts[strategy] += 1
                    else:
                        strategy_counts[strategy] = 1
                
                # Sort by count
                sorted_strategies = sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)
                segment_strategies[segment] = sorted_strategies
            
            # Store results
            self.analysis_results['retention_strategies'] = {
                'segment_strategies': segment_strategies,
                'customer_strategies': self.results_df[['customer_id', 'segment_name', 'retention_strategies']].to_dict('records')
            }
            
            return self.analysis_results['retention_strategies']
            
        except Exception as e:
            return {'error': f'Error in generating retention strategies: {str(e)}'}
    
    def _get_customer_strategies(self, row, base_strategies, segment_top_reasons, reason_strategies):
        """Generate strategies for a specific customer
        
        Args:
            row (Series): Customer data row
            base_strategies (dict): Base strategies by segment
            segment_top_reasons (dict): Top reasons by segment
            reason_strategies (dict): Strategies by reason
            
        Returns:
            list: Personalized strategies for the customer
        """
        segment = row['segment_name']
        strategies = []
        
        # Add base strategies for the segment
        if segment in base_strategies:
            strategies.extend(base_strategies[segment])
        
        # Add strategies based on customer's specific reasons
        if 'top_reasons' in row and reason_strategies:
            for reason in row['top_reasons']:
                if reason in reason_strategies:
                    # Add the first strategy for this reason
                    if reason_strategies[reason]:
                        strategies.append(reason_strategies[reason][0])
        
        # Ensure we don't have duplicates
        strategies = list(set(strategies))
        
        # Limit to top 5 strategies
        return strategies[:5]
    
    def generate_roi_analysis(self):
        """Generate ROI analysis for retention strategies
        
        Returns:
            dict: ROI analysis results
        """
        if self.results_df is None or 'retention_strategies' not in self.results_df.columns:
            return {'error': 'Retention strategies must be generated first'}
        
        try:
            # Assume average customer lifetime value and retention costs
            avg_clv = 1000  # Average customer lifetime value
            
            # Define cost and effectiveness for each strategy type
            strategy_metrics = {
                'Loyalty rewards program': {'cost': 50, 'effectiveness': 0.7},
                'Early access to new features': {'cost': 20, 'effectiveness': 0.5},
                'Referral incentives': {'cost': 100, 'effectiveness': 0.6},
                'Proactive customer service check-ins': {'cost': 30, 'effectiveness': 0.6},
                'Targeted discounts on current services': {'cost': 150, 'effectiveness': 0.7},
                'Service upgrades at reduced cost': {'cost': 200, 'effectiveness': 0.8},
                'Immediate outreach by customer success team': {'cost': 100, 'effectiveness': 0.5},
                'Substantial contract renewal incentives': {'cost': 300, 'effectiveness': 0.8},
                'Personalized retention offers based on usage patterns': {'cost': 250, 'effectiveness': 0.75},
                'Offer 15% discount for 12-month commitment': {'cost': 180, 'effectiveness': 0.7},
                'Introduce tiered pricing plan': {'cost': 50, 'effectiveness': 0.6},
                'Bundle services for cost savings': {'cost': 100, 'effectiveness': 0.65},
                'Highlight benefits of 12-month contracts': {'cost': 20, 'effectiveness': 0.4},
                'Offer one-time bonus for contract signup': {'cost': 150, 'effectiveness': 0.7},
                'Create flexible contract options': {'cost': 50, 'effectiveness': 0.5},
                'Competitive price matching': {'cost': 200, 'effectiveness': 0.75},
                'Highlight unique value propositions': {'cost': 30, 'effectiveness': 0.5},
                'Offer exclusive features not available with competitors': {'cost': 100, 'effectiveness': 0.6},
                'Immediate service quality review': {'cost': 80, 'effectiveness': 0.6},
                'Dedicated support representative': {'cost': 150, 'effectiveness': 0.7},
                'Service credits for past issues': {'cost': 120, 'effectiveness': 0.65},
                'Free trial of premium features': {'cost': 50, 'effectiveness': 0.55},
                'Customized feature package': {'cost': 100, 'effectiveness': 0.6},
                'Early access to beta features': {'cost': 30, 'effectiveness': 0.45},
                'Priority technical support': {'cost': 80, 'effectiveness': 0.6},
                'System audit and optimization': {'cost': 200, 'effectiveness': 0.7},
                'Automatic monitoring and issue resolution': {'cost': 150, 'effectiveness': 0.65}
            }
            
            # Calculate ROI for each customer
            def calculate_customer_roi(row):
                if not isinstance(row['retention_strategies'], list) or not row['retention_strategies']:
                    return {'cost': 0, 'effectiveness': 0, 'roi': 0}
                
                # Calculate total cost and effectiveness
                total_cost = 0
                total_effectiveness = 0
                strategies_used = 0
                
                for strategy in row['retention_strategies']:
                    if strategy in strategy_metrics:
                        total_cost += strategy_metrics[strategy]['cost']
                        total_effectiveness += strategy_metrics[strategy]['effectiveness']
                        strategies_used += 1
                
                # Calculate average effectiveness
                avg_effectiveness = total_effectiveness / strategies_used if strategies_used > 0 else 0
                
                # Calculate expected value saved
                churn_prob = row['churn_probability'] if 'churn_probability' in row else 0.5
                expected_value_saved = avg_clv * churn_prob * avg_effectiveness
                
                # Calculate ROI
                roi = (expected_value_saved - total_cost) / total_cost if total_cost > 0 else 0
                
                return {
                    'cost': total_cost,
                    'effectiveness': avg_effectiveness,
                    'expected_value_saved': expected_value_saved,
                    'roi': roi
                }
            
            # Apply ROI calculation
            self.results_df['roi_analysis'] = self.results_df.apply(calculate_customer_roi, axis=1)
            
            # Extract ROI components
            self.results_df['strategy_cost'] = self.results_df['roi_analysis'].apply(lambda x: x['cost'])
            self.results_df['strategy_effectiveness'] = self.results_df['roi_analysis'].apply(lambda x: x['effectiveness'])
            self.results_df['expected_value_saved'] = self.results_df['roi_analysis'].apply(lambda x: x.get('expected_value_saved', 0))
            self.results_df['strategy_roi'] = self.results_df['roi_analysis'].apply(lambda x: x['roi'])
            
            # Calculate segment-level ROI
            segment_roi = {}
            for segment in self.results_df['segment_name'].unique():
                segment_df = self.results_df[self.results_df['segment_name'] == segment]
                
                segment_roi[segment] = {
                    'avg_cost': segment_df['strategy_cost'].mean(),
                    'avg_effectiveness': segment_df['strategy_effectiveness'].mean(),
                    'avg_expected_value_saved': segment_df['expected_value_saved'].mean(),
                    'avg_roi': segment_df['strategy_roi'].mean(),
                    'total_cost': segment_df['strategy_cost'].sum(),
                    'total_expected_value_saved': segment_df['expected_value_saved'].sum(),
                    'net_value': segment_df['expected_value_saved'].sum() - segment_df['strategy_cost'].sum()
                }
            
            # Calculate overall ROI
            overall_roi = {
                'total_cost': self.results_df['strategy_cost'].sum(),
                'total_expected_value_saved': self.results_df['expected_value_saved'].sum(),
                'net_value': self.results_df['expected_value_saved'].sum() - self.results_df['strategy_cost'].sum(),
                'overall_roi': (self.results_df['expected_value_saved'].sum() - self.results_df['strategy_cost'].sum()) / 
                              self.results_df['strategy_cost'].sum() if self.results_df['strategy_cost'].sum() > 0 else 0
            }
            
            # Store results
            self.analysis_results['roi_analysis'] = {
                'segment_roi': segment_roi,
                'overall_roi': overall_roi,
                'customer_roi': self.results_df[['customer_id', 'segment_name', 'strategy_cost', 
                                               'strategy_effectiveness', 'expected_value_saved', 'strategy_roi']].to_dict('records')
            }
            
            return self.analysis_results['roi_analysis']
            
        except Exception as e:
            return {'error': f'Error in ROI analysis: {str(e)}'}
    
    def run_complete_analysis(self, results_df=None):
        """Run all analysis steps
        
        Args:
            results_df (DataFrame, optional): DataFrame with prediction results
            
        Returns:
            dict: Complete analysis results
        """
        if results_df is not None:
            self.load_data(results_df)
        
        if self.results_df is None:
            return {'error': 'No data loaded'}
        
        # Run all analyses
        self.segment_customers()
        self.analyze_churn_factors()
        self.generate_retention_strategies()
        self.generate_roi_analysis()
        
        # Return all results
        return self.analysis_results
    
    def get_customer_insights(self, customer_id):
        """Get detailed insights for a specific customer
        
        Args:
            customer_id: ID of the customer
            
        Returns:
            dict: Customer insights
        """
        if self.results_df is None:
            return {'error': 'No data loaded'}
        
        try:
            # Find the customer
            customer_df = self.results_df[self.results_df['customer_id'] == customer_id]
            if len(customer_df) == 0:
                return {'error': f'Customer {customer_id} not found'}
            
            # Get customer data
            customer_data = customer_df.iloc[0].to_dict()
            
            # Create insights
            insights = {
                'customer_id': customer_id,
                'churn_probability': customer_data.get('churn_probability', 0),
                'risk_category': customer_data.get('risk_category', 'Unknown'),
                'segment': customer_data.get('segment_name', 'Unknown'),
                'top_reasons': customer_data.get('top_reasons', []),
                'retention_strategies': customer_data.get('retention_strategies', []),
                'roi_analysis': customer_data.get('roi_analysis', {})
            }
            
            # Add comparison to segment averages if segmentation was performed
            if 'segment_name' in customer_data and 'segmentation' in self.analysis_results:
                segment = customer_data['segment_name']
                segment_stats = self.analysis_results['segmentation']['cluster_stats']
                
                # Convert segment stats to regular dict if it's a nested dict
                if isinstance(segment_stats, dict) and 'churn_probability_mean' in segment_stats:
                    segment_avg_churn = segment_stats['churn_probability_mean'].get(customer_data['customer_segment'], 0)
                else:
                    # Try to find the right segment in the stats
                    segment_avg_churn = 0
                    for key, value in segment_stats.items():
                        if isinstance(value, dict) and 'churn_probability_mean' in value:
                            segment_avg_churn = value['churn_probability_mean']
                            break
                
                insights['segment_comparison'] = {
                    'churn_probability_vs_segment': customer_data.get('churn_probability', 0) - segment_avg_churn,
                    'segment_avg_churn': segment_avg_churn
                }
            
            return insights
            
        except Exception as e:
            return {'error': f'Error getting customer insights: {str(e)}'}

# Example usage
def example_usage():
    # Generate sample prediction results
    from models.churn_model import generate_synthetic_data, ChurnModel
    
    # Generate data
    df = generate_synthetic_data(1000)
    
    # Create and train model
    model = ChurnModel(model_type='logistic')
    model.train(df)
    
    # Make predictions
    results_df = model.predict(df)
    
    # Create analyzer and run analysis
    analyzer = ChurnAnalyzer(results_df)
    analysis_results = analyzer.run_complete_analysis()
    
    # Print sample results
    print("\nCustomer Segmentation:")
    print(f"Number of segments: {len(analysis_results['segmentation']['segment_mapping'])}")
    print(f"Segment mapping: {analysis_results['segmentation']['segment_mapping']}")
    
    print("\nChurn Factors Analysis:")
    print(f"Top reasons: {analysis_results['churn_factors']['sorted_reasons'][:3]}")
    
    print("\nRetention Strategies:")
    for segment, strategies in analysis_results['retention_strategies']['segment_strategies'].items():
        print(f"\nTop strategies for {segment}:")
        for strategy, count in strategies[:3]:
            print(f"  - {strategy}: {count}")
    
    print("\nROI Analysis:")
    print(f"Overall ROI: {analysis_results['roi_analysis']['overall_roi']['overall_roi']:.2f}")
    print(f"Total cost: ${analysis_results['roi_analysis']['overall_roi']['total_cost']:.2f}")
    print(f"Total expected value saved: ${analysis_results['roi_analysis']['overall_roi']['total_expected_value_saved']:.2f}")
    print(f"Net value: ${analysis_results['roi_analysis']['overall_roi']['net_value']:.2f}")

if __name__ == "__main__":
    example_usage()