import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import 'katex/dist/katex.min.css';
import {
  ChevronRight,
  ChevronDown,
  Eye,
  Globe,
  Brain,
  Database,
  Layers,
  Activity,
  ArrowLeft,
  Menu,
  X
} from 'lucide-react';

// Documentation content structure
const docsStructure = {
  'getting-started': {
    title: 'Getting Started',
    icon: <Activity size={18} />,
    pages: {
      'introduction': {
        title: 'Introduction',
        content: `
# Introduction to EDEN

EDEN (Emotionally-Driven Embodied Navigation) is a humanoid robotics framework designed for adaptive reasoning, emotional context awareness, and long-term memory.

## What is EDEN?

EDEN represents a paradigm shift in how robots interact with humans. Instead of treating each interaction as an isolated event, EDEN maintains a continuous understanding of:

- **User relationships** and past interactions
- **Emotional context** and social dynamics
- **Long-term goals** and behavioral patterns

## Key Features

1. **Adaptive Cognition** - Decisions evolve based on experience
2. **Supermemory** - Persistent knowledge graph for context
3. **Real-time Processing** - Edge computing on Jetson platforms
4. **ROS 2 Integration** - Industry-standard robotics middleware

## Quick Start

\`\`\`bash
# Clone the repository
git clone https://github.com/EDEN-robotics/Eden.git

# Install dependencies
cd Eden && pip install -r requirements.txt

# Run the cognitive loop
python main.py --config config/default.yaml
\`\`\`
        `
      },
      'installation': {
        title: 'Installation',
        content: `
# Installation Guide

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| GPU | CUDA 11.0 | CUDA 12.0+ |
| Storage | 20 GB | 50+ GB SSD |

## Hardware Setup

### Jetson Nano Configuration

The Jetson Nano handles real-time perception and motor control:

\`\`\`bash
# Flash JetPack 5.1
sudo apt update && sudo apt upgrade
sudo apt install nvidia-jetpack

# Install ROS 2 Humble
sudo apt install ros-humble-desktop
\`\`\`

### Host PC Setup

The host PC runs the cognitive and planning layers:

\`\`\`bash
# Create virtual environment
python -m venv eden_env
source eden_env/bin/activate

# Install EDEN
pip install eden-robotics
\`\`\`

## Verification

Run the diagnostic tool to verify installation:

\`\`\`bash
eden-diagnose --all
\`\`\`
        `
      },
      'architecture-overview': {
        title: 'Architecture Overview',
        content: `
# System Architecture

EDEN uses a distributed multi-processing pipeline connecting local perception to remote cognition.

## High-Level Overview

\`\`\`
┌─────────────────────────────────────────────────────────────┐
│                      EDEN Architecture                       │
├─────────────────────────────────────────────────────────────┤
│  Jetson Nano (Edge)          │    Host PC (Cloud/Local)    │
│  ├── Perception Layer        │    ├── Cognitive Layer      │
│  ├── Input Processing        │    ├── Supermemory          │
│  └── Motor Control           │    └── Planning Layer       │
└─────────────────────────────────────────────────────────────┘
\`\`\`

## Data Flow

The cognitive loop processes information through five distinct layers:

1. **Input Layer** → Sensory data collection
2. **Context Layer** → Situational awareness
3. **Cognitive Layer** → Reasoning and evaluation
4. **Planning Layer** → Action generation
5. **Action Layer** → Motor execution
        `
      }
    }
  },
  'perception-layer': {
    title: 'Perception Layer',
    icon: <Eye size={18} />,
    pages: {
      'overview': {
        title: 'Overview',
        content: `
# Perception Layer

The Perception Layer is responsible for processing all sensory inputs from the robot's environment.

## Core Components

### Vision System

EDEN uses a multi-camera setup for comprehensive environment understanding:

- **RGB Cameras** - High-resolution color imaging
- **Depth Sensors** - Intel RealSense D455 for 3D perception
- **IR Cameras** - Low-light operation capability

### Human Detection Pipeline

The human detection system uses a cascade of models:

\`\`\`python
class HumanDetector:
    def __init__(self):
        self.pose_model = MediaPipePose()
        self.face_model = FaceRecognition()
        self.action_model = ActionRecognizer()
    
    def process_frame(self, frame):
        poses = self.pose_model.detect(frame)
        faces = self.face_model.identify(frame)
        actions = self.action_model.classify(frame, poses)
        return PerceptionOutput(poses, faces, actions)
\`\`\`

## Mathematical Foundation

### Image Processing

The raw image $I \\in \\mathbb{R}^{H \\times W \\times 3}$ is processed through:

$$
F = \\text{CNN}(I; \\theta)
$$

where $F \\in \\mathbb{R}^{h \\times w \\times d}$ is the feature map and $\\theta$ represents learned parameters.

### Object Detection

Bounding boxes are predicted using:

$$
\\hat{b} = \\sigma(W_b \\cdot F + b_b)
$$

with confidence scores:

$$
p(\\text{object}) = \\sigma(W_c \\cdot F + b_c)
$$
        `
      },
      'vision-processing': {
        title: 'Vision Processing',
        content: `
# Vision Processing

## Camera Calibration

Intrinsic camera matrix:

$$
K = \\begin{bmatrix}
f_x & 0 & c_x \\\\
0 & f_y & c_y \\\\
0 & 0 & 1
\\end{bmatrix}
$$

## Depth Estimation

Stereo depth is computed using:

$$
Z = \\frac{f \\cdot B}{d}
$$

where:
- $Z$ is the depth
- $f$ is the focal length
- $B$ is the baseline between cameras
- $d$ is the disparity

## Point Cloud Generation

3D points are reconstructed from depth:

$$
\\begin{bmatrix} X \\\\ Y \\\\ Z \\end{bmatrix} = Z \\cdot K^{-1} \\begin{bmatrix} u \\\\ v \\\\ 1 \\end{bmatrix}
$$

## Implementation

\`\`\`python
import numpy as np

def depth_to_pointcloud(depth, K):
    """Convert depth image to 3D point cloud."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Homogeneous coordinates
    uv1 = np.stack([u, v, np.ones_like(u)], axis=-1)
    
    # Back-project to 3D
    K_inv = np.linalg.inv(K)
    xyz = depth[..., None] * (uv1 @ K_inv.T)
    
    return xyz
\`\`\`
        `
      },
      'action-recognition': {
        title: 'Action Recognition',
        content: `
# Human Action Recognition

## Temporal Modeling

Actions are recognized using temporal convolutional networks:

$$
h_t = \\text{TCN}(x_{t-k:t})
$$

where $x_{t-k:t}$ represents a window of $k$ frames.

## Skeleton-Based Recognition

Pose keypoints are extracted and processed:

\`\`\`python
class SkeletonActionModel:
    def __init__(self, num_classes=15):
        self.gcn = SpatioTemporalGCN()
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, skeleton_sequence):
        # skeleton_sequence: (B, T, V, C)
        # B=batch, T=frames, V=joints, C=coordinates
        features = self.gcn(skeleton_sequence)
        return self.classifier(features)
\`\`\`

## Supported Actions

| Action | Description | Confidence Threshold |
|--------|-------------|---------------------|
| Walking | Locomotion toward robot | 0.85 |
| Waving | Greeting gesture | 0.90 |
| Pointing | Directional indication | 0.80 |
| Sitting | Seated posture | 0.95 |
| Reaching | Object interaction | 0.75 |
        `
      }
    }
  },
  'context-layer': {
    title: 'Context Layer',
    icon: <Globe size={18} />,
    pages: {
      'overview': {
        title: 'Overview',
        content: `
# Context Layer

The Context Layer gathers and maintains situational awareness about the environment and users.

## Purpose

While the Perception Layer answers "What do I see?", the Context Layer answers:

- **Who** is the user?
- **What** is their emotional state?
- **Where** is this interaction happening?
- **When** did we last interact?

## Context Vector

The context is represented as a high-dimensional vector:

$$
\\mathbf{c} = [\\mathbf{u}, \\mathbf{e}, \\mathbf{s}, \\mathbf{h}]
$$

where:
- $\\mathbf{u}$ = User identity embedding
- $\\mathbf{e}$ = Emotional state vector
- $\\mathbf{s}$ = Spatial context
- $\\mathbf{h}$ = Historical interaction summary
        `
      },
      'user-identification': {
        title: 'User Identification',
        content: `
# User Identification

## Face Embedding

Users are identified using face embeddings:

$$
\\mathbf{e}_{\\text{face}} = \\text{FaceNet}(I_{\\text{crop}})
$$

## Matching

Identity is determined by nearest neighbor in embedding space:

$$
\\text{id} = \\arg\\min_i \\|\\mathbf{e}_{\\text{face}} - \\mathbf{e}_i\\|_2
$$

with verification threshold $\\tau = 0.6$.

\`\`\`python
class UserIdentifier:
    def __init__(self, database_path):
        self.embeddings = self.load_database(database_path)
        self.threshold = 0.6
    
    def identify(self, face_embedding):
        distances = np.linalg.norm(
            self.embeddings - face_embedding, 
            axis=1
        )
        min_idx = np.argmin(distances)
        
        if distances[min_idx] < self.threshold:
            return self.user_ids[min_idx]
        return "unknown"
\`\`\`
        `
      },
      'emotional-analysis': {
        title: 'Emotional Analysis',
        content: `
# Emotional Analysis

## Multi-Modal Emotion Recognition

Emotions are inferred from multiple signals:

### Facial Expression Analysis

Using a 7-class emotion classifier:

$$
p(\\text{emotion}|\\text{face}) = \\text{softmax}(W \\cdot \\mathbf{e}_{\\text{face}})
$$

### Voice Prosody Analysis

Audio features are extracted using:

$$
\\mathbf{a} = [\\text{pitch}, \\text{energy}, \\text{MFCC}_{1:13}]
$$

### Fusion

Final emotional state combines modalities:

$$
\\mathbf{e}_{\\text{final}} = \\alpha \\cdot \\mathbf{e}_{\\text{visual}} + (1-\\alpha) \\cdot \\mathbf{e}_{\\text{audio}}
$$

where $\\alpha$ is learned based on signal quality.

## Emotion Categories

| Primary | Secondary | Valence | Arousal |
|---------|-----------|---------|---------|
| Happy | Excited | +0.8 | +0.7 |
| Sad | Disappointed | -0.7 | -0.3 |
| Angry | Frustrated | -0.6 | +0.8 |
| Neutral | Calm | 0.0 | -0.2 |
        `
      }
    }
  },
  'cognitive-layer': {
    title: 'Cognitive Layer',
    icon: <Brain size={18} />,
    pages: {
      'overview': {
        title: 'Overview',
        content: `
# Cognitive Layer

The Cognitive Layer is the "brain" of EDEN, performing reasoning and decision-making based on perception and context.

## Architecture

\`\`\`
┌─────────────────────────────────────────┐
│           Cognitive Layer               │
├─────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ Working │  │ Reason- │  │ Goal    │ │
│  │ Memory  │→ │ ing     │→ │ Eval    │ │
│  └─────────┘  └─────────┘  └─────────┘ │
│       ↑            ↓            ↓       │
│  ┌─────────────────────────────────┐   │
│  │        Supermemory Query        │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
\`\`\`

## Core Functions

1. **State Evaluation** - Assess current situation
2. **Goal Comparison** - Compare with desired outcomes
3. **Action Selection** - Choose optimal response
4. **Learning** - Update beliefs based on outcomes
        `
      },
      'reasoning-engine': {
        title: 'Reasoning Engine',
        content: `
# Reasoning Engine

## Probabilistic Reasoning

EDEN uses a probabilistic framework for decision-making:

$$
P(A|S, G) = \\frac{P(G|A, S) \\cdot P(A|S)}{P(G|S)}
$$

where:
- $A$ = Action
- $S$ = Current state
- $G$ = Goal state

## Utility Function

Actions are scored using expected utility:

$$
U(a) = \\sum_{s'} P(s'|s, a) \\cdot R(s, a, s')
$$

## Implementation

\`\`\`python
class ReasoningEngine:
    def __init__(self, memory):
        self.memory = memory
        self.policy = PolicyNetwork()
    
    def evaluate_action(self, state, action, goal):
        # Query similar past situations
        past_outcomes = self.memory.query(state, action)
        
        # Estimate probability of success
        p_success = self.estimate_success(past_outcomes)
        
        # Calculate expected utility
        utility = p_success * self.reward(goal) 
        utility -= (1 - p_success) * self.penalty(goal)
        
        return utility
\`\`\`
        `
      },
      'decision-making': {
        title: 'Decision Making',
        content: `
# Decision Making

## Action Selection

The optimal action maximizes expected utility:

$$
a^* = \\arg\\max_a \\mathbb{E}[U(s, a) + \\gamma V(s')]
$$

## Exploration vs Exploitation

EDEN balances exploration using softmax selection:

$$
P(a|s) = \\frac{\\exp(Q(s,a)/\\tau)}{\\sum_{a'} \\exp(Q(s,a')/\\tau)}
$$

where $\\tau$ is the temperature parameter controlling randomness.

## Social Constraints

Actions are filtered through social acceptability:

$$
a_{\\text{valid}} = \\{a : C_{\\text{social}}(a, s) > \\theta_{\\text{social}}\\}
$$

This ensures the robot never takes socially inappropriate actions, regardless of task utility.
        `
      }
    }
  },
  'supermemory': {
    title: 'Supermemory',
    icon: <Database size={18} />,
    pages: {
      'overview': {
        title: 'Overview',
        content: `
# Supermemory System

Supermemory is EDEN's unique long-term memory architecture that enables continuous learning and personalization.

## Why Supermemory?

Traditional robots lack memory continuity. Supermemory provides:

- **Persistent knowledge** across sessions
- **User relationship modeling** over time
- **Experience-based decision refinement**

## Architecture

\`\`\`
┌────────────────────────────────────────────────┐
│                 Supermemory                     │
├────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐         │
│  │   Vector     │    │   Graph      │         │
│  │   Store      │←──→│   Database   │         │
│  └──────────────┘    └──────────────┘         │
│         ↑                   ↑                  │
│  ┌──────────────────────────────────┐         │
│  │      Memory Controller           │         │
│  └──────────────────────────────────┘         │
└────────────────────────────────────────────────┘
\`\`\`
        `
      },
      'knowledge-graph': {
        title: 'Knowledge Graph',
        content: `
# Knowledge Graph

## Structure

The knowledge graph stores entities and relationships:

$$
G = (V, E, \\phi_V, \\phi_E)
$$

where:
- $V$ = Set of entities (users, objects, events)
- $E$ = Set of relationships
- $\\phi_V$ = Entity embeddings
- $\\phi_E$ = Relationship embeddings

## Entity Types

| Type | Description | Example |
|------|-------------|---------|
| User | Human individuals | "John Smith" |
| Object | Physical items | "Red mug" |
| Location | Spatial regions | "Kitchen" |
| Event | Temporal occurrences | "Morning greeting" |
| Concept | Abstract ideas | "Happiness" |

## Relationship Queries

\`\`\`python
# Find all positive interactions with user
query = """
MATCH (u:User {name: $user_name})-[r:INTERACTED]->(e:Event)
WHERE r.sentiment > 0.5
RETURN e.timestamp, e.type, r.sentiment
ORDER BY e.timestamp DESC
LIMIT 10
"""
\`\`\`
        `
      },
      'memory-consolidation': {
        title: 'Memory Consolidation',
        content: `
# Memory Consolidation

## Short-term to Long-term Transfer

Memories are consolidated based on importance:

$$
I(m) = \\alpha \\cdot R(m) + \\beta \\cdot N(m) + \\gamma \\cdot E(m)
$$

where:
- $R(m)$ = Recency
- $N(m)$ = Novelty
- $E(m)$ = Emotional intensity

## Forgetting Curve

Low-importance memories decay:

$$
S(t) = S_0 \\cdot e^{-t/\\tau}
$$

## Retrieval

Memories are retrieved using similarity search:

$$
\\text{memories} = \\text{top}_k\\{m : \\cos(\\mathbf{q}, \\mathbf{m}) > \\theta\\}
$$

\`\`\`python
class MemoryRetrieval:
    def query(self, context, k=5):
        query_embedding = self.encoder(context)
        
        # Vector similarity search
        similarities = self.index.search(query_embedding, k)
        
        # Re-rank by recency and importance
        results = self.rerank(similarities, context)
        
        return results
\`\`\`
        `
      }
    }
  },
  'planning-layer': {
    title: 'Planning Layer',
    icon: <Layers size={18} />,
    pages: {
      'overview': {
        title: 'Overview',
        content: `
# Planning Layer

The Planning Layer translates cognitive decisions into executable motion plans.

## Responsibilities

1. **Path Planning** - Collision-free navigation
2. **Motion Generation** - Smooth trajectories
3. **Social Compliance** - Human-aware movement
4. **Real-time Adaptation** - Dynamic replanning

## Planning Pipeline

\`\`\`
Cognitive Decision → Goal Specification → Path Planning 
    → Trajectory Optimization → Motion Commands
\`\`\`
        `
      },
      'trajectory-planning': {
        title: 'Trajectory Planning',
        content: `
# Trajectory Planning

## Cost Function

Trajectories minimize a composite cost:

$$
J(\\xi) = \\int_0^T \\left[ c_{\\text{smooth}}(\\xi) + c_{\\text{obstacle}}(\\xi) + c_{\\text{social}}(\\xi) \\right] dt
$$

## Smoothness Cost

Minimizes jerk for natural motion:

$$
c_{\\text{smooth}} = \\|\\dddot{q}\\|^2
$$

## Social Cost

Maintains appropriate distance from humans:

$$
c_{\\text{social}} = \\sum_i \\exp\\left(-\\frac{\\|x - h_i\\|^2}{2\\sigma^2}\\right)
$$

## Optimization

\`\`\`python
class TrajectoryOptimizer:
    def optimize(self, start, goal, obstacles, humans):
        # Initialize with straight-line path
        xi = self.initial_trajectory(start, goal)
        
        for _ in range(self.max_iters):
            # Compute gradients
            grad = self.compute_gradient(xi, obstacles, humans)
            
            # Update trajectory
            xi = xi - self.learning_rate * grad
            
            # Project to feasible space
            xi = self.project_constraints(xi)
        
        return xi
\`\`\`
        `
      },
      'motion-primitives': {
        title: 'Motion Primitives',
        content: `
# Motion Primitives

## Definition

Motion primitives are reusable building blocks:

$$
\\mathcal{P} = \\{p_1, p_2, ..., p_n\\}
$$

Each primitive $p_i$ is a parameterized trajectory:

$$
p_i(t; \\theta) : [0, 1] \\rightarrow \\mathcal{C}
$$

## Primitive Library

| Primitive | Parameters | Use Case |
|-----------|------------|----------|
| Reach | target, speed | Object interaction |
| Wave | amplitude, frequency | Greeting |
| Point | direction | Drawing attention |
| Nod | angle | Acknowledgment |
| Turn | angle, pivot | Reorientation |

## Sequencing

Complex behaviors combine primitives:

\`\`\`python
class BehaviorSequencer:
    def greet_user(self, user_position):
        return Sequence([
            TurnToward(user_position),
            Wave(amplitude=0.3, duration=1.5),
            Speak("Hello!"),
            WaitForResponse(timeout=3.0)
        ])
\`\`\`
        `
      }
    }
  },
  'action-layer': {
    title: 'Action Layer',
    icon: <Activity size={18} />,
    pages: {
      'overview': {
        title: 'Overview',
        content: `
# Action Layer

The Action Layer executes planned motions on physical hardware through ROS 2.

## Components

1. **Motor Controllers** - Joint-level control
2. **Sensor Integration** - Real-time feedback
3. **Safety System** - Emergency stops and limits
4. **Communication** - WebSocket bridge to Host PC

## ROS 2 Architecture

\`\`\`
┌─────────────────────────────────────────────┐
│              ROS 2 Node Graph               │
├─────────────────────────────────────────────┤
│  /eden_bridge ←→ /motion_controller         │
│       ↓              ↓                      │
│  /joint_states   /joint_commands            │
│       ↓              ↓                      │
│  /safety_node    /hardware_interface        │
└─────────────────────────────────────────────┘
\`\`\`
        `
      },
      'ros2-integration': {
        title: 'ROS 2 Integration',
        content: `
# ROS 2 Integration

## Node Structure

\`\`\`python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

class EdenBridge(Node):
    def __init__(self):
        super().__init__('eden_bridge')
        
        # Publishers
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/command',
            10
        )
        
        # Subscribers
        self.state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.state_callback,
            10
        )
        
        # WebSocket to Host PC
        self.ws_client = WebSocketClient(HOST_PC_URL)
    
    def execute_trajectory(self, trajectory):
        msg = self.convert_to_ros(trajectory)
        self.trajectory_pub.publish(msg)
\`\`\`

## Message Types

| Topic | Type | Direction | Rate |
|-------|------|-----------|------|
| /joint_states | JointState | Robot → Bridge | 100 Hz |
| /joint_commands | JointTrajectory | Bridge → Robot | 50 Hz |
| /eden/perception | PerceptionMsg | Jetson → Host | 30 Hz |
| /eden/action | ActionMsg | Host → Jetson | 10 Hz |
        `
      },
      'safety-systems': {
        title: 'Safety Systems',
        content: `
# Safety Systems

## Multi-Layer Safety

EDEN implements defense-in-depth:

### Layer 1: Software Limits

\`\`\`python
class JointLimits:
    POSITION_LIMITS = {
        'shoulder': (-180, 180),  # degrees
        'elbow': (0, 150),
        'wrist': (-90, 90)
    }
    
    VELOCITY_LIMITS = {
        'shoulder': 60,  # deg/s
        'elbow': 90,
        'wrist': 120
    }
    
    TORQUE_LIMITS = {
        'shoulder': 50,  # Nm
        'elbow': 30,
        'wrist': 10
    }
\`\`\`

### Layer 2: Collision Avoidance

Real-time obstacle detection triggers slowdown:

$$
v_{\\text{safe}} = v_{\\text{max}} \\cdot \\min\\left(1, \\frac{d - d_{\\text{stop}}}{d_{\\text{slow}} - d_{\\text{stop}}}\\right)
$$

### Layer 3: Hardware E-Stop

Physical emergency stop button immediately cuts power to all actuators.

## Safety Metrics

| Metric | Threshold | Action |
|--------|-----------|--------|
| Human proximity | < 0.5m | Reduce speed 50% |
| Torque spike | > 1.5x nominal | Immediate stop |
| Communication loss | > 100ms | Hold position |
        `
      }
    }
  }
};

// Flatten structure for easy lookup
const flattenDocs = (structure) => {
  const flat = {};
  Object.entries(structure).forEach(([folder, data]) => {
    Object.entries(data.pages).forEach(([page, content]) => {
      const key = folder + '/' + page;
      flat[key] = {
        ...content,
        folder,
        folderTitle: data.title
      };
    });
  });
  return flat;
};

const flatDocs = flattenDocs(docsStructure);

// Sidebar Component
const Sidebar = ({ currentPath, isOpen, onClose }) => {
  const [expandedFolders, setExpandedFolders] = useState(
    Object.keys(docsStructure).reduce((acc, key) => ({ ...acc, [key]: true }), {})
  );

  const toggleFolder = (folder) => {
    setExpandedFolders(prev => ({ ...prev, [folder]: !prev[folder] }));
  };

  const sidebarClasses = [
    'fixed lg:sticky top-0 left-0 z-50 lg:z-auto',
    'w-72 h-screen bg-bg-primary border-r border-white/10',
    'transform transition-transform duration-300 ease-in-out',
    isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0',
    'overflow-y-auto'
  ].join(' ');

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={onClose}
        />
      )}

      <aside className={sidebarClasses}>
        {/* Header */}
        <div className="sticky top-0 bg-bg-primary border-b border-white/10 p-4">
          <div className="flex items-center justify-between">
            <Link to="/" className="flex items-center gap-2 text-white hover:text-accent transition-colors">
              <ArrowLeft size={18} />
              <span className="font-mono text-sm">Back to Home</span>
            </Link>
            <button
              onClick={onClose}
              className="lg:hidden p-2 hover:bg-white/10 rounded-lg"
            >
              <X size={20} />
            </button>
          </div>
          <h1 className="text-xl font-bold mt-4 gradient-text">Documentation</h1>
        </div>

        {/* Navigation */}
        <nav className="p-4">
          {Object.entries(docsStructure).map(([folderKey, folder]) => (
            <div key={folderKey} className="mb-2">
              <button
                onClick={() => toggleFolder(folderKey)}
                className="w-full flex items-center gap-2 p-2 rounded-lg hover:bg-white/5 transition-colors text-left"
              >
                {expandedFolders[folderKey] ? (
                  <ChevronDown size={16} className="text-text-dim" />
                ) : (
                  <ChevronRight size={16} className="text-text-dim" />
                )}
                <span className="text-accent">{folder.icon}</span>
                <span className="font-medium">{folder.title}</span>
              </button>

              {expandedFolders[folderKey] && (
                <div className="ml-6 mt-1 space-y-1">
                  {Object.entries(folder.pages).map(([pageKey, page]) => {
                    const path = folderKey + '/' + pageKey;
                    const isActive = currentPath === path;

                    const linkClasses = [
                      'block p-2 pl-4 rounded-lg text-sm transition-colors border-l-2',
                      isActive
                        ? 'bg-accent/10 border-accent text-white'
                        : 'border-transparent text-text-dim hover:bg-white/5 hover:text-white'
                    ].join(' ');

                    return (
                      <Link
                        key={pageKey}
                        to={'/docs/' + path}
                        onClick={onClose}
                        className={linkClasses}
                      >
                        {page.title}
                      </Link>
                    );
                  })}
                </div>
              )}
            </div>
          ))}
        </nav>
      </aside>
    </>
  );
};

// Main Documentation Component
export default function Documentation() {
  const { '*': path } = useParams();
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Default to first page if no path
  const currentPath = path || 'getting-started/introduction';
  const currentDoc = flatDocs[currentPath];

  useEffect(() => {
    if (!path) {
      navigate('/docs/getting-started/introduction', { replace: true });
    }
  }, [path, navigate]);

  if (!currentDoc) {
    return (
      <div className="min-h-screen bg-bg-primary flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-4xl font-bold mb-4">Page Not Found</h1>
          <Link to="/docs/getting-started/introduction" className="text-accent hover:underline">
            Go to Documentation Home
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-bg-primary flex">
      <Sidebar
        currentPath={currentPath}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      {/* Main Content */}
      <main className="flex-1 min-w-0">
        {/* Mobile header */}
        <div className="sticky top-0 z-30 lg:hidden bg-bg-primary border-b border-white/10 p-4">
          <button
            onClick={() => setSidebarOpen(true)}
            className="flex items-center gap-2 text-white"
          >
            <Menu size={24} />
            <span className="font-medium">Menu</span>
          </button>
        </div>

        {/* Content */}
        <article className="max-w-4xl mx-auto px-6 py-12">
          {/* Breadcrumb */}
          <div className="flex items-center gap-2 text-sm text-text-dim mb-8">
            <Link to="/docs" className="hover:text-white transition-colors">Docs</Link>
            <ChevronRight size={14} />
            <span>{currentDoc.folderTitle}</span>
            <ChevronRight size={14} />
            <span className="text-white">{currentDoc.title}</span>
          </div>

          {/* Markdown Content */}
          <div className="prose prose-invert prose-lg max-w-none">
            <ReactMarkdown
              remarkPlugins={[remarkMath, remarkGfm]}
              rehypePlugins={[rehypeKatex, rehypeHighlight]}
              components={{
                h1: ({ children }) => (
                  <h1 className="text-4xl font-bold gradient-text mb-8">{children}</h1>
                ),
                h2: ({ children }) => (
                  <h2 className="text-2xl font-bold mt-12 mb-4 pb-2 border-b border-white/10">{children}</h2>
                ),
                h3: ({ children }) => (
                  <h3 className="text-xl font-semibold mt-8 mb-3">{children}</h3>
                ),
                p: ({ children }) => (
                  <p className="text-text-dim leading-relaxed mb-4">{children}</p>
                ),
                ul: ({ children }) => (
                  <ul className="list-disc list-inside space-y-2 text-text-dim mb-4">{children}</ul>
                ),
                ol: ({ children }) => (
                  <ol className="list-decimal list-inside space-y-2 text-text-dim mb-4">{children}</ol>
                ),
                li: ({ children }) => (
                  <li className="text-text-dim">{children}</li>
                ),
                code: ({ inline, className, children }) => {
                  if (inline) {
                    return (
                      <code className="bg-white/10 px-2 py-0.5 rounded text-accent font-mono text-sm">
                        {children}
                      </code>
                    );
                  }
                  return (
                    <code className={className}>
                      {children}
                    </code>
                  );
                },
                pre: ({ children }) => (
                  <pre className="bg-black/50 border border-white/10 rounded-xl p-4 overflow-x-auto mb-6">
                    {children}
                  </pre>
                ),
                table: ({ children }) => (
                  <div className="overflow-x-auto mb-6">
                    <table className="w-full border-collapse border border-white/10 rounded-xl overflow-hidden">
                      {children}
                    </table>
                  </div>
                ),
                thead: ({ children }) => (
                  <thead className="bg-white/5">{children}</thead>
                ),
                th: ({ children }) => (
                  <th className="border border-white/10 px-4 py-2 text-left font-semibold">{children}</th>
                ),
                td: ({ children }) => (
                  <td className="border border-white/10 px-4 py-2 text-text-dim">{children}</td>
                ),
                blockquote: ({ children }) => (
                  <blockquote className="border-l-4 border-accent pl-4 italic text-text-dim my-4">
                    {children}
                  </blockquote>
                ),
                a: ({ href, children }) => (
                  <a href={href} className="text-accent hover:underline" target="_blank" rel="noopener noreferrer">
                    {children}
                  </a>
                ),
                img: ({ src, alt }) => (
                  <img
                    src={src}
                    alt={alt}
                    className="rounded-xl border border-white/10 my-6 max-w-full"
                  />
                )
              }}
            >
              {currentDoc.content}
            </ReactMarkdown>
          </div>

          {/* Navigation */}
          <div className="flex justify-between mt-16 pt-8 border-t border-white/10">
            {/* Previous / Next navigation could be added here */}
          </div>
        </article>
      </main>
    </div>
  );
}
