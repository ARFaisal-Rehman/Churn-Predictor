package com.riskradar.service;

import java.util.HashMap;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;
import java.util.Random;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 * ChurnPredictionService - Java service for advanced churn prediction functionality
 * 
 * This service can be integrated with the Python Flask application using py4j or similar
 * Java-Python bridge technologies. It provides additional prediction capabilities and
 * performance optimizations for large datasets.
 */
public class ChurnPredictionService {
    
    private Random random = new Random(42); // For reproducible results in demo
    
    /**
     * Predicts churn probability for a customer based on their attributes
     * 
     * @param customerAttributes Map of customer attribute names to values
     * @return Churn probability between 0 and 1
     */
    public double predictChurnProbability(Map<String, Object> customerAttributes) {
        // In a real implementation, this would use a trained model
        // For demo purposes, we'll generate a weighted random score
        
        double baseScore = random.nextDouble();
        
        // Apply some simple rules to adjust the score based on known churn factors
        if (customerAttributes.containsKey("tenure")) {
            int tenure = Integer.parseInt(customerAttributes.get("tenure").toString());
            if (tenure < 12) {
                baseScore += 0.2; // New customers are more likely to churn
            } else {
                baseScore -= 0.1; // Loyal customers are less likely to churn
            }
        }
        
        if (customerAttributes.containsKey("contract")) {
            String contract = customerAttributes.get("contract").toString();
            if (contract.equalsIgnoreCase("Month-to-month")) {
                baseScore += 0.15; // Month-to-month customers are more likely to churn
            } else if (contract.equalsIgnoreCase("Two year")) {
                baseScore -= 0.2; // Long-term contract customers are less likely to churn
            }
        }
        
        if (customerAttributes.containsKey("monthly_charges")) {
            double monthlyCharges = Double.parseDouble(customerAttributes.get("monthly_charges").toString());
            if (monthlyCharges > 100) {
                baseScore += 0.1; // Higher charges increase churn risk
            }
        }
        
        // Ensure the score is between 0 and 1
        return Math.max(0, Math.min(1, baseScore));
    }
    
    /**
     * Batch processes a CSV file to generate churn predictions
     * 
     * @param filePath Path to the CSV file
     * @return List of maps containing customer data with added churn predictions
     */
    public List<Map<String, Object>> batchProcessFile(String filePath) throws IOException {
        List<Map<String, Object>> results = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        
        // Read header
        String line = reader.readLine();
        String[] headers = line.split(",");
        
        // Process each line
        while ((line = reader.readLine()) != null) {
            String[] values = line.split(",");
            Map<String, Object> customer = new HashMap<>();
            
            // Parse customer data
            for (int i = 0; i < headers.length; i++) {
                if (i < values.length) {
                    customer.put(headers[i], values[i]);
                }
            }
            
            // Add prediction
            double churnProbability = predictChurnProbability(customer);
            customer.put("churn_probability", churnProbability);
            
            // Add risk category
            if (churnProbability < 0.3) {
                customer.put("risk_category", "Low");
            } else if (churnProbability < 0.7) {
                customer.put("risk_category", "Medium");
            } else {
                customer.put("risk_category", "High");
            }
            
            // Add to results
            results.add(customer);
        }
        
        reader.close();
        return results;
    }
    
    /**
     * Generates retention offers based on customer data and churn risk
     * 
     * @param customerData Map of customer attribute names to values
     * @param riskCategory Risk category (Low, Medium, High)
     * @return Map containing offer details
     */
    public Map<String, Object> generateRetentionOffer(Map<String, Object> customerData, String riskCategory) {
        Map<String, Object> offer = new HashMap<>();
        
        // Base offer details
        offer.put("offer_id", "OFF-" + System.currentTimeMillis());
        offer.put("created_date", System.currentTimeMillis());
        offer.put("customer_id", customerData.get("customer_id"));
        offer.put("risk_category", riskCategory);
        
        // Generate offer based on risk category
        switch (riskCategory) {
            case "High":
                offer.put("discount_percentage", 20);
                offer.put("discount_duration_months", 12);
                offer.put("premium_features", true);
                offer.put("description", "20% discount for 12 months + premium features");
                break;
                
            case "Medium":
                offer.put("discount_percentage", 10);
                offer.put("discount_duration_months", 6);
                offer.put("premium_features", false);
                offer.put("description", "10% discount for 6 months");
                break;
                
            case "Low":
            default:
                offer.put("discount_percentage", 0);
                offer.put("discount_duration_months", 0);
                offer.put("premium_features", false);
                offer.put("description", "Customer satisfaction survey");
                break;
        }
        
        // Customize based on customer attributes
        if (customerData.containsKey("contract") && 
            customerData.get("contract").toString().equalsIgnoreCase("Month-to-month")) {
            offer.put("contract_offer", "One year contract with first month free");
        }
        
        if (customerData.containsKey("internet_service") && 
            customerData.get("internet_service").toString().equalsIgnoreCase("Fiber optic")) {
            offer.put("service_upgrade", "Free speed boost for 3 months");
        }
        
        return offer;
    }
    
    /**
     * Identifies the top reasons for potential churn based on customer attributes
     * 
     * @param customerData Map of customer attribute names to values
     * @return List of reasons for potential churn, ordered by importance
     */
    public List<String> identifyChurnReasons(Map<String, Object> customerData) {
        List<String> reasons = new ArrayList<>();
        
        // Check for common churn factors
        if (customerData.containsKey("contract") && 
            customerData.get("contract").toString().equalsIgnoreCase("Month-to-month")) {
            reasons.add("No long-term contract");
        }
        
        if (customerData.containsKey("monthly_charges")) {
            double monthlyCharges = Double.parseDouble(customerData.get("monthly_charges").toString());
            if (monthlyCharges > 100) {
                reasons.add("High monthly charges");
            }
        }
        
        if (customerData.containsKey("tenure")) {
            int tenure = Integer.parseInt(customerData.get("tenure").toString());
            if (tenure < 12) {
                reasons.add("New customer (less than 1 year)");
            }
        }
        
        if (customerData.containsKey("tech_support") && 
            customerData.get("tech_support").toString().equalsIgnoreCase("No")) {
            reasons.add("No tech support");
        }
        
        if (customerData.containsKey("online_security") && 
            customerData.get("online_security").toString().equalsIgnoreCase("No")) {
            reasons.add("No online security");
        }
        
        if (customerData.containsKey("payment_method") && 
            customerData.get("payment_method").toString().contains("electronic")) {
            reasons.add("Electronic payment method");
        }
        
        // If we don't have enough reasons, add some generic ones
        if (reasons.size() < 3) {
            List<String> genericReasons = new ArrayList<>();
            genericReasons.add("Competitor offers");
            genericReasons.add("Service dissatisfaction");
            genericReasons.add("Lack of features");
            genericReasons.add("Technical issues");
            genericReasons.add("Price sensitivity");
            
            // Add random generic reasons until we have at least 3
            while (reasons.size() < 3 && !genericReasons.isEmpty()) {
                int index = random.nextInt(genericReasons.size());
                reasons.add(genericReasons.get(index));
                genericReasons.remove(index);
            }
        }
        
        return reasons;
    }
    
    /**
     * Main method for testing the service
     */
    public static void main(String[] args) {
        ChurnPredictionService service = new ChurnPredictionService();
        
        // Test with a sample customer
        Map<String, Object> customer = new HashMap<>();
        customer.put("customer_id", "CUST001");
        customer.put("tenure", 5);
        customer.put("contract", "Month-to-month");
        customer.put("monthly_charges", 120.45);
        customer.put("tech_support", "No");
        
        double probability = service.predictChurnProbability(customer);
        System.out.println("Churn Probability: " + probability);
        
        String riskCategory = probability < 0.3 ? "Low" : (probability < 0.7 ? "Medium" : "High");
        System.out.println("Risk Category: " + riskCategory);
        
        List<String> reasons = service.identifyChurnReasons(customer);
        System.out.println("Churn Reasons: " + reasons);
        
        Map<String, Object> offer = service.generateRetentionOffer(customer, riskCategory);
        System.out.println("Retention Offer: " + offer.get("description"));
    }
}