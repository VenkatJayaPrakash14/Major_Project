import google.generativeai as genai
from typing import List, Dict
import os
from markupsafe import Markup

class GeminiHelper:
    def __init__(self, api_key: str):
        """Initialize the Gemini helper with API key and model configuration."""
        if not api_key:
            raise ValueError("API key is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def get_crop_insights(self, crops: List[str], soil_params: Dict[str, float]) -> str:
        """
        Generate detailed insights about recommended crops using Gemini Pro.
        
        Args:
            crops: List of recommended crop names
            soil_params: Dictionary containing soil and environmental parameters
            
        Returns:
            str: HTML-formatted insights about the crops
        """
        if not crops:
            return "<p>No crops provided for analysis.</p>"
        
        prompt = f"""
        Analyze these recommended crops as an agricultural expert: {', '.join(crops)}
        
        Current Growing Conditions:
        - Nitrogen (N): {soil_params['N']:.1f} mg/kg
        - Phosphorus (P): {soil_params['P']:.1f} mg/kg
        - Potassium (K): {soil_params['K']:.1f} mg/kg
        - Temperature: {soil_params['temperature']:.1f}°C
        - Humidity: {soil_params['humidity']:.1f}%
        - pH Level: {soil_params['ph']:.1f}
        - Rainfall: {soil_params['rainfall']:.1f} mm
        
        Provide a detailed analysis in HTML format with the following sections:
        
        1. Crop-Soil Compatibility
        - Explain how the current soil conditions match each crop's requirements
        - Highlight any optimal or suboptimal parameters
        
        2. Growing Guidelines
        - Specific planting techniques for these soil conditions
        - Irrigation recommendations based on the rainfall and humidity levels
        - Fertilization strategy considering current NPK levels
        
        3. Environmental Considerations
        - How temperature and humidity affect these crops
        - Seasonal timing recommendations
        - Climate-specific adaptations
        
        4. Risk Factors & Solutions
        - Potential challenges given the current conditions
        - Preventive measures and solutions
        - Monitoring recommendations
        
        Format your response using these HTML elements only:
        <h3> for section headings
        <p> for paragraphs
        <ul> and <li> for lists
        <strong> for emphasis
        
        Keep the tone professional and practical.
        """
        
        try:
            response = self.model.generate_content(prompt)
            
            # Process and structure the response
            formatted_response = response.text
            
            # Ensure the response starts with a container div
            formatted_response = f'''
            <div class="crop-insights">
                {formatted_response}
            </div>
            '''
            
            # Return as a safe markup string
            return Markup(formatted_response)
            
        except Exception as e:
            return Markup(f'''
            <div class="error-message bg-red-50 p-4 rounded-lg">
                <p class="text-red-600">Error generating insights: {str(e)}</p>
            </div>
            ''')