import subprocess
from flask import Flask, request, send_file

app = Flask(__name__)

@app.route('/')
def index():
    return send_file('process_table.html')

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

remote_computer = "pooji"
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
            /* Add any custom CSS styles here */
        </style>
    </head>
    <body>
        {html_table}
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

if __name__ == '__main__':
    app.run(port=5003,debug=True)