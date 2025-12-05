# What-If Analysis Dashboard

A comprehensive project optimization tool with AI-powered resource allocation and interactive constraint adjustment.

## Features

### 🎯 **Core Functionality**
- **Process Selection**: Choose from Hospital, Software, or Manufacturing projects
- **AI Optimization**: Get the best scenario using advanced algorithms
- **Resource Allocation**: Detailed breakdown of task assignments
- **Real-time Impact**: See cost/time changes as you adjust constraints

### 🛠️ **Constraint Adjustment**

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
cd api
python main.py
```

The API will be available at `http://localhost:8002`

## API Endpoints

- `GET /` - Health check
- `POST /optimize/{process_name}` - Optimize standard processes
- `POST /optimize/cms-process/{process_id}` - Optimize CMS processes
- `POST /optimize/custom` - Optimize with custom constraints
- `GET /processes` - List available processes

## Deployment

For Railway deployment:
1. Connect your GitHub repository to Railway
2. Railway will auto-detect Python and deploy using railway.json
3. Set environment variables as needed

## Project Structure

```
backend/
├── api/
│   ├── main.py              # FastAPI application
│   └── requirements.txt     # API dependencies
├── src/
│   ├── models/             # Data models and transformers
│   ├── optimization/       # Optimization algorithms
│   ├── agents/            # ML agents
│   └── environment/       # Simulation environment
├── example/               # Sample project data
├── railway.json           # Railway deployment config
├── main.py               # Alternative entry point
└── requirements.txt       # Main dependencies

### **Resource Constraints**
- Hourly rates ($30-$200)
- Max hours per day (4-12 hours)
- Resource availability (on/off)

### **Task Constraints**
- Task duration (5-100 hours)
- Task priority (Very Low to Critical)
- Parallel execution allowance

### **Optimization Preferences**
- Time priority (0-100%)
- Cost priority (0-100%)
- Quality priority (0-100%)

## Example Scenarios

### Hospital Project
- 8 tasks from patient intake to hospital-wide rollout
- 8 specialized healthcare resources
- Focus on clinical operations and compliance

### Software Project
- 6 tasks from database design to testing
- 6 developers with different specializations
- Modern e-commerce platform development

### Manufacturing Project
- 5 tasks for smart factory implementation
- 5 engineering specialists
- IoT and automation focus

## Technical Details

- **Backend**: FastAPI with async support
- **Frontend**: Vanilla JavaScript with modern CSS
- **Optimization**: Multiple algorithms (Pareto, RL, heuristic)
- **Real-time Updates**: Live constraint impact calculation
- **Responsive Design**: Works on desktop and mobile

Start exploring different "what-if" scenarios to optimize your projects!
