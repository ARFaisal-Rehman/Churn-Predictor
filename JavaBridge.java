// JavaBridge.java - Bridge class for connecting Java service with Python via py4j

package com.riskradar.churn;

import java.util.*;
import java.io.*;

/**
 * JavaBridge serves as the entry point for py4j to access Java functionality
 * from the Python Flask application.
 */
public class JavaBridge {
    private ChurnPredictionService predictionService;
    
    public JavaBridge() {
        this.predictionService = new ChurnPredictionService();
        System.out.println("JavaBridge initialized with ChurnPredictionService");
    }
    
    /**
     * Predicts churn probability for a single customer
     * 
     * @param customerData Map containing customer attributes
     * @return Map containing prediction results
     */
    public Map<String, Object> predictChurn(Map<String, Object> customerData) {
        try {
            return predictionService.predictChurn(customerData);
        } catch (Exception e) {
            System.err.println("Error in JavaBridge.predictChurn: " + e.getMessage());
            e.printStackTrace();
            
            // Return error information
            Map<String, Object> errorResult = new HashMap<>();
            errorResult.put("error", true);
            errorResult.put("message", e.getMessage());
            return errorResult;
        }
    }
    
    /**
     * Process a batch of customers for churn prediction
     * 
     * @param customersData List of Maps containing customer attributes
     * @return List of Maps containing prediction results
     */
    public List<Map<String, Object>> batchPredictChurn(List<Map<String, Object>> customersData) {
        try {
            return predictionService.batchPredictChurn(customersData);
        } catch (Exception e) {
            System.err.println("Error in JavaBridge.batchPredictChurn: " + e.getMessage());
            e.printStackTrace();
            
            // Return error information
            List<Map<String, Object>> errorResults = new ArrayList<>();
            Map<String, Object> errorResult = new HashMap<>();
            errorResult.put("error", true);
            errorResult.put("message", e.getMessage());
            errorResults.add(errorResult);
            return errorResults;
        }
    }
    
    /**
     * Generate retention offers for customers
     * 
     * @param customerData Map containing customer attributes and churn probability
     * @return Map containing retention offer details
     */
    public Map<String, Object> generateRetentionOffer(Map<String, Object> customerData) {
        try {
            return predictionService.generateRetentionOffer(customerData);
        } catch (Exception e) {
            System.err.println("Error in JavaBridge.generateRetentionOffer: " + e.getMessage());
            e.printStackTrace();
            
            // Return error information
            Map<String, Object> errorResult = new HashMap<>();
            errorResult.put("error", true);
            errorResult.put("message", e.getMessage());
            return errorResult;
        }
    }
    
    /**
     * Process a CSV file for batch prediction
     * 
     * @param csvFilePath Path to the CSV file
     * @param outputFilePath Path where the output CSV should be saved
     * @return boolean indicating success or failure
     */
    public boolean processCsvFile(String csvFilePath, String outputFilePath) {
        try {
            return predictionService.processCsvFile(csvFilePath, outputFilePath);
        } catch (Exception e) {
            System.err.println("Error in JavaBridge.processCsvFile: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }
    
    /**
     * Get the top reasons for churn for a specific customer
     * 
     * @param customerData Map containing customer attributes
     * @return List of reasons for potential churn
     */
    public List<String> getChurnReasons(Map<String, Object> customerData) {
        try {
            return predictionService.identifyChurnReasons(customerData);
        } catch (Exception e) {
            System.err.println("Error in JavaBridge.getChurnReasons: " + e.getMessage());
            e.printStackTrace();
            
            // Return error information
            List<String> errorResults = new ArrayList<>();
            errorResults.add("Error: " + e.getMessage());
            return errorResults;
        }
    }
    
    /**
     * Simple test method to verify the bridge is working
     * 
     * @return String confirmation message
     */
    public String testConnection() {
        return "JavaBridge connection successful. ChurnPredictionService is ready.";
    }
}