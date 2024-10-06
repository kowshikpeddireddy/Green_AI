from flask import Flask, request, render_template, redirect, url_for,send_file,render_template_string
from PIL import Image
import os
import subprocess
import pandas as pd
import torchvision.transforms as transforms
import torch
from model_definition import LigResNeXt
import re
import threading
import google.generativeai as genai
from google.generativeai.types import SafetySettingDict

app = Flask(__name__, static_folder='static')

genai.configure(api_key="AIzaSyDhYcCutSjIBaKK6-YeY3xVyIZzLbq9yrI")
gemini_model = genai.GenerativeModel('gemini-pro')
 
model = LigResNeXt.load_from_checkpoint('epoch=14-val_loss=0.13508102297782898.ckpt')

model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
def recommend_alternative(predicted_label):
    alternative_images = {
        'battery': {
            'name': 'Rechargeable Batteries',
            'url': 'download.jpeg',
            'reason': 'Using rechargeable batteries instead of disposable ones can significantly reduce carbon emissions. Rechargeable batteries are recyclable and can help reduce waste.',
            'CO2 Emissions': 'Depends on usage, but approximately 50 grams per battery',
            'Recyclable': 'Yes',
            'Strengths': 'Can be reused multiple times, reducing waste and environmental impact.'
        },
        'biological': {
            'url': 'alternative_biological_image.jpg',
            'reason': 'Composting food scraps reduces methane emissions from landfills and enriches soil. Biological waste is biodegradable and can be turned into nutrient-rich compost.',
            'CO2 Emissions': 'Approximately 20 grams per kilogram of composted waste',
            'Recyclable': 'N/A',
            'Strengths': 'Supports soil health and reduces methane emissions from landfills.'
        },
        'brown-glass': {
            'url': 'alternative_brown_glass_image.jpg',
            'reason': 'Choosing products packaged in brown glass reduces the need for virgin materials and energy in manufacturing. Brown glass is widely recyclable.',
            'CO2 Emissions': 'Approximately 50 grams per kilogram of brown glass produced',
            'Recyclable': 'Yes',
            'Strengths': 'Reduces the demand for new materials and energy in glass production.'
        },
        'cardboard': {
            'url': 'alternative_cardboard_image.jpg',
            'reason': 'Opting for products with minimal packaging or recyclable cardboard packaging reduces waste and carbon emissions. Cardboard is biodegradable and widely recyclable.',
            'CO2 Emissions': 'Approximately 30 grams per kilogram of cardboard produced',
            'Recyclable': 'Yes',
            'Strengths': 'Lightweight and versatile packaging material that can be recycled.'
        },
        'clothes': {
            'url': 'alternative_clothes_image.jpg',
            'reason': 'Buying second-hand clothing or supporting sustainable fashion brands reduces the environmental impact of clothing production.',
            'CO2 Emissions': 'Varies widely depending on fabric and production methods, but approximately 15-20 kilograms per kilogram of clothing',
            'Recyclable': 'N/A',
            'Strengths': 'Reduces demand for new clothing production and supports sustainable fashion practices.'
        },
        'green-glass': {
            'url': 'alternative_green_glass_image.jpg',
            'reason': 'Choosing products packaged in green glass reduces the need for virgin materials and energy in manufacturing. Green glass is widely recyclable.',
            'CO2 Emissions': 'Approximately 40 grams per kilogram of green glass produced',
            'Recyclable': 'Yes',
            'Strengths': 'Reduces the demand for new materials and energy in glass production.'
        },
        'metal': {
            'name': 'Aluminium',
            'url': 'alu.png',
            'reason': 'Aluminum is highly recyclable, and recycling it requires significantly less energy compared to primary production.',
            'CO2 Emissions': 'Approximately 10-20 kilograms per kilogram of aluminum produced (from primary production), significantly less from recycling',
            'Recyclable': 'Yes',
            'Strengths': 'Highly durable and can be recycled indefinitely with minimal loss of quality.'
        },
        'paper': {
            'url': 'alternative_paper_image.jpg',
            'reason': 'Using recycled paper products reduces the demand for new trees and energy in paper production. Paper is biodegradable and widely recyclable.',
            'CO2 Emissions': 'Approximately 2.5 kilograms per kilogram of paper produced',
            'Recyclable': 'Yes',
            'Strengths': 'Renewable resource that can be recycled into new paper products.'
        },
        'plastic': {
            'name': 'Paper Bags',
            'url': 'img.png',
            'reason': 'Choosing biodegradable or reusable alternatives to single-use paper bags reduces pollution and carbon emissions.',
            'CO2 Emissions': 'Approximately 6 kilograms per kilogram of plastic produced (varies greatly depending on type and production method)',
            'Recyclable': 'Varies (biodegradable options are recyclable)',
            'Strengths': 'Biodegradable options break down naturally, reducing environmental impact.'
        },
        'shoes': {
            'url': 'alternative_shoes_image.jpg',
            'reason': 'Opting for sustainably made shoes or repairing old ones instead of buying new ones reduces waste and carbon emissions.',
            'CO2 Emissions': 'Varies widely depending on materials and production methods, but approximately 20-30 kilograms per kilogram of shoes',
            'Recyclable': 'N/A',
            'Strengths': 'Supports sustainable manufacturing practices and reduces demand for new shoe production.'
        },
        'trash': {
            'url': 'alternative_trash_image.jpg',
            'reason': 'Segregating waste for recycling and composting reduces the amount of waste sent to landfills and incinerators, reducing carbon emissions.',
            'CO2 Emissions': 'Depends on composition, but approximately 1-2 kilograms per kilogram of trash sent to landfill or incinerator',
            'Recyclable': 'N/A',
            'Strengths': 'Reduces landfill waste and supports waste management strategies.'
        },
        'white-glass': {
            'name': 'PTEG',
            'url': 'pteg.jpg',
            'reason': 'Glass manufacturing using PTEG can be more sustainable and reduce emissions. When melted, no additional CO2 is released.',
            'CO2 Emissions': 'Approximately 45 grams per kilogram of white glass produced',
            'Recyclable': 'Yes',
            'Strengths': 'Reduces the environmental impact of glass production and can be recycled.'
        }
        # Add more labels and their alternative images and reasons as needed
    }
    return alternative_images.get(predicted_label, None)


class_labels = {
    0: 'battery',
    1: 'biological',
    2: 'brown-glass',
    3: 'cardboard',
    4: 'clothes',
    5: 'green-glass',
    6: 'metal',
    7: 'paper',
    8: 'plastic',
    9: 'shoes',
    10: 'trash',
    11: 'white-glass'
}
def predict_label(image):
    # Apply transformations
    img = transform(image).unsqueeze(0)
    # Predict
    with torch.no_grad():
        output = model(img)
    # Get predicted class index
    pred_index = torch.argmax(output).item()
    # Map the index to label
    predicted_label = class_labels[pred_index]
    return predicted_label
def read_emission_csv():
    df = pd.read_csv( 'emission.csv' )
    # Extract the required columns
    time = df['start_time']
    power_consumption = df['power_consumption(kWh)']
    co2_emissions = df['CO2_emissions(kg)']
    cpu_name = df['CPU_name']
    os = df['OS']
    region = df['region/country']
    return time, power_consumption, co2_emissions, cpu_name, os, region
@app.route('/')
def index():
    return render_template('mian.html')
@app.route('/material', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file:
            # Save the uploaded file to the 'uploads' directory
            file_path = os.path.join( 'static', file.filename )
            file.save(file_path)

            # Read image from file
            img = Image.open(file_path).convert("RGB")
            # Predict label
            predicted_label = predict_label(img)
            # Redirect to the result page
            return redirect(url_for('result', filename=file.filename, predicted_label=predicted_label))

    return render_template('material.html')


@app.route('/result/<filename>/<predicted_label>')
def result(filename, predicted_label):
    alternative = recommend_alternative(predicted_label)
    return render_template('result.html', filename=filename, predicted_label=predicted_label, alternative=alternative)


@app.route('/machine')
def run_emission_tracker():
    # Provide the complete path to the 'emission_tracker.py' file
    #emission_process = subprocess.Popen(['python', r'C:\Users\Kowshik\PycharmProjects\ECO to AI - Copy\eco2ai\emission_track.py'])
    # Read the 'emission.csv' file and extract required details
    time, power_consumption, co2_emissions, cpu_name, os, region = read_emission_csv()
    # Pass the extracted values to the HTML template
    return render_template('emissions.html', time=time, power_consumption=power_consumption,
                           co2_emissions=co2_emissions, cpu_name=cpu_name, os=os, region=region)
"""
def calculate_distance(source, destination):
    try:
        # Get coordinates of source and destination
        source_location = geolocator.geocode(source)
        destination_location = geolocator.geocode(destination)

        # Calculate distance between the two locations
        distance = geodesic((source_location.latitude, source_location.longitude),
                            (destination_location.latitude, destination_location.longitude)).kilometers

        return distance
    except:
        return None

"""
def calculate_emissions(distance, days):
    if distance is None:
        return None
    else:
        train_emission = distance * 0.04 * days  # kg CO2 per km for train
        flight_emission = distance * 0.2 * days  # kg CO2 per km for flight
        car_emission = distance * 0.15 * days  # kg CO2 per km for car

        return {
            'train': train_emission,
            'flight': flight_emission,
            'car': car_emission
        }

def get_transport_suggestion(emissions):
    min_emission = min(emissions.values())
    suggestion = [mode for mode, emission in emissions.items() if emission == min_emission]
    return suggestion[0] if suggestion else None

@app.route('/trip', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        source_name = request.form['source']
        destination_name = request.form['destination']
        days = int(request.form['days'])

        try:
            response = gemini_model.generate_content(f"Calculate the distance between {source_name} and {destination_name} in kilometers")
            distance_text = response.text
            distance = re.search(r'\b\d+\b', distance_text)
            if distance:
                distance = int(distance.group())
                print(distance)
                emissions = calculate_emissions(distance, days)
                best_route = get_transport_suggestion(emissions)
                if distance > 1000:
                    redirect_url = f"https://www.makemytrip.com/flight/search?itinerary={source_name}-{destination_name}-{'YYYYMMDD'}"
                    return render_template('trip.html', source=source_name, destination=destination_name, distance=distance, emissions=emissions, best_route='flight', days=days)
                else:
                    return render_template('trip.html', source=source_name, destination=destination_name, distance=distance, emissions=emissions, best_route=best_route, days=days)
            else:
                return "Error: Unable to extract distance from the response"
        except Exception as e:
            print(f"Error: {e}")
            return render_template('trip.html')
    return render_template('trip.html')


@app.route ( '/calculate' )
def calculate():
    return render_template ( 'calculate.html' )
@app.route('/logs')
def indo():
    return send_file('process_table.html')
@app.route('/electricity')
def ind():
    return render_template ('electricity.html')
@app.route('/terminate/<pid>', methods=['POST'])
def terminate_process(pid):
    try:
        # Code to terminate the process with the given PID
        # You may need to adjust this part depending on the platform (Windows, Linux, etc.)
        # For Windows, you can use taskkill command
        result = subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True, text=True, check=True)
        return 'Process terminated successfully', 200
    except subprocess.CalledProcessError as e:
        return f'Error terminating process: {e.stderr}', 500
    except Exception as e:
        return f'An error occurred: {str(e)}', 500

remote_computer = "LAPTOP-EQU6SNBM"
command = f"Get-Process -ComputerName {remote_computer} | Sort-Object -Property CPU -Descending | Select-Object Handles, NPM, PM, WS, CPU, Id, SI, ProcessName, @{{Name='CO2 Emissions'; Expression={{ $_.CPU * 0.02 }}}}"

# Construct PowerShell command
ps_command = f'powershell.exe -Command "{command}"'

try:
    # Execute the command
    process = subprocess.run(ps_command, capture_output=True, text=True, check=True)

    # Parse the output
    output_lines = process.stdout.strip().split('\n')
    table_data = []

    for line in output_lines:
        parts = line.split(":", 1)
        if len(parts) == 2:
            table_data.append(parts[1].strip())

    # Chunk the data into rows with 9 columns each
    table_rows = [table_data[i:i+9] for i in range(0, len(table_data), 9)]

    # Generate HTML table
    html_table = "<table>"
    html_table += "<thead><tr><th>Handles</th><th>NPM</th><th>PM</th><th>WS</th><th>CPU</th><th>Id</th><th>SI</th><th>ProcessName</th><th>CO2 Emissions</th><th></th></tr></thead>"
    html_table += "<tbody>"
    for row in table_rows:
        html_table += "<tr>"
        for i, cell in enumerate(row):
            if i == 4 and cell and float(cell) > 3:  # Check if CPU is non-empty and greater than 3
                html_table += f"<td>{cell} <button class='cancel-btn' data-pid='{row[5]}'>Cancel</button></td>"
            else:
                html_table += f"<td>{cell}</td>"
        html_table += "</tr>"
    html_table += "</tbody></table>"

    # Complete HTML document with table and JavaScript
    html_page = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Process Table</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f0f2f5;
                color: #333;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                max-width: 1000px;
                margin: auto;
                background: #fff;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}
            h1 {{
                text-align: center;
                color: #333;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            table, th, td {{
                border: 1px solid #ddd;
            }}
            th, td {{
                text-align: left;
                padding: 8px;
            }}
            th {{
                background-color: #007bff;
                color: #ffffff;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .cancel-btn {{
                display: inline-block;
                padding: 5px 10px;
                background-color: #dc3545;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }}
            .cancel-btn:hover {{
                background-color: #c82333;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Process Table</h1>
            {html_table}
        </div>
        <script>
            document.querySelectorAll('.cancel-btn').forEach(item => {{
                item.addEventListener('click', event => {{
                    let pid = item.getAttribute('data-pid');
                    fetch('/terminate/' + pid, {{ method: 'POST' }})
                        .then(response => {{
                            if (!response.ok) {{
                                throw new Error('Network response was not ok');
                            }}
                            return response.text();
                        }})
                        .then(data => {{
                            console.log(data);
                            // Optionally, you can reload the page or update the table here
                        }})
                        .catch(error => {{
                            console.error('Error:', error.message);
                            alert('Error terminating process');
                        }});
                }});
            }});
        </script>
    </body>
    </html>
    """

    # Save the HTML content to a file
    with open("process_table.html", "w") as html_file:
        html_file.write(html_page)

except subprocess.CalledProcessError as e:
    print(f"Error executing PowerShell command: {e.stderr}")
except Exception as e:
    print(f"An error occurred: {str(e)}")



@app.route('/code', methods=['GET', 'POST'])
def inde():
    if request.method == 'POST':
        user_code = request.form['user_code']
        if user_code.strip () != '':
            optimized_code = optimize_code ( user_code )
            return render_template_string (
                """
                <html>
                <head>
                    <title>Optimized Code</title>
                    <style>
                        body {
                            font-family: Times New Roman;
                            background-color: #f5f5f5;
                            font-weight:bold;
                            font-size:16px;
                            color: #333;
                            text-align: center;
                        }
                        h1 {
                            margin-top: 50px;
                            color: #007bff;
                        }
                        pre {
                            padding: 20px;
                            background-color: #f8f9fa;
                            border-radius: 8px;
                            max-width: 800px;
                            margin: 0 auto;
                            overflow-x: auto;
                        }
                        form {
                            margin-top: 30px;
                        }
                        textarea {
                            width: 80%;
                            height: 200px;
                            padding: 10px;
                            border: 1px solid #ced4da;
                            border-radius: 4px;
                            resize: none;
                        }
                        button {
                            background-color: #28a745;
                            color: white;
                            padding: 10px 20px;
                            border: none;
                            border-radius: 4px;
                            cursor: pointer;
                            transition: background-color 0.3s ease;
                        }
                        button:hover {
                            background-color: #218838;
                        }
                    </style>
                </head>
                <body>
                    <h1>Optimized Code</h1>
                    <pre>{{ optimized_code }}</pre>
                    <form method="post">
                        <textarea name="user_code" rows="10" cols="50">{{ user_code }}</textarea><br>
                        <button type="submit">Optimize Again</button>
                    </form>
                </body>
                </html>
                """,
                optimized_code=optimized_code,
                user_code=user_code
            )
    return render_template_string (
        """
        <html>
        <head>
            <title>Enter Python Code</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f5f5f5;
                    color: #333;
                    text-align: center;
                }
                h1 {
                    margin-top: 50px;
                    color: #007bff;
                }
                form {
                    margin-top: 30px;
                }
                textarea {
                    width: 80%;
                    height: 200px;
                    padding: 10px;
                    border: 1px solid #ced4da;
                    border-radius: 4px;
                    resize: none;
                }
                button {
                    background-color: #007bff;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    transition: background-color 0.3s ease;
                }
                button:hover {
                    background-color: #0056b3;
                }
            </style>
        </head>
        <body>
            <h1>Enter Python Code</h1>
            <form method="post">
                <textarea name="user_code" rows="10" cols="50"></textarea><br>
                <button type="submit">Optimize</button>
            </form>
        </body>
        </html>
        """
    )


def optimize_code(user_code):
    if not user_code.strip():
        return "Error: User code is empty or contains only whitespace."

    try:
        response = gemini_model.generate_content(f'optimize and give the new code: {user_code}')
        if response.text:
            return response.text
        else:
            return "Error: Empty response received from API"
    except Exception as e:
        return f"Error during API request: {e}"


@app.route('/chatbot')
def chatbot():
    try:
        # Call the chatbot code as a subprocess
        subprocess.Popen(['python', 'chatbot.py'])
        return render_template('chatbot.html')
    except Exception as e:
        return "Error launching chatbot: " + str(e)
if __name__ == '__main__':
    emission_process = subprocess.Popen (
        ['python', r'emission_track.py'] )
    #os.makedirs( 'uploads', exist_ok=True )
    # Serve static files from the 'uploads' directory
    app.static_folder = 'static'
    # Run the Flask application
    app.run(debug=True)