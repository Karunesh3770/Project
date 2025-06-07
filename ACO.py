import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import math
import random
from PIL import Image
import io

# --- ACO Class with animation support ---
class ACO:
    def __init__(self, n_ants: int, n_iterations: int, decay: float, alpha: float, beta: float):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.ant_paths_per_iter = []  # For animation
        self.best_path_per_iter = []  # For animation
        self.best_distance_per_iter = []
    
    def initialize_pheromone(self, n_locations: int) -> np.ndarray:
        return np.ones((n_locations, n_locations)) * 0.1
    
    def calculate_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    def create_distance_matrix(self, locations: List[Tuple[float, float]]) -> np.ndarray:
        n_locations = len(locations)
        distance_matrix = np.zeros((n_locations, n_locations))
        for i in range(n_locations):
            for j in range(n_locations):
                if i != j:
                    distance_matrix[i][j] = self.calculate_distance(locations[i], locations[j])
        return distance_matrix
    
    def select_next_location(self, current_loc: int, unvisited: List[int], pheromone: np.ndarray, distance_matrix: np.ndarray) -> int:
        probabilities = []
        for loc in unvisited:
            tau = pheromone[current_loc][loc]
            eta = 1.0 / distance_matrix[current_loc][loc]
            probability = (tau ** self.alpha) * (eta ** self.beta)
            probabilities.append((loc, probability))
        total = sum(p for _, p in probabilities)
        probabilities = [(loc, p/total) for loc, p in probabilities]
        locations, probs = zip(*probabilities)
        return np.random.choice(locations, p=probs)
    
    def run(self, locations: List[Tuple[float, float]]):
        n_locations = len(locations)
        distance_matrix = self.create_distance_matrix(locations)
        pheromone = self.initialize_pheromone(n_locations)
        best_path = None
        best_distance = float('inf')
        self.ant_paths_per_iter = []
        self.best_path_per_iter = []
        self.best_distance_per_iter = []
        for iteration in range(self.n_iterations):
            ant_paths = []
            for ant in range(self.n_ants):
                current_loc = random.randint(0, n_locations - 1)
                unvisited = list(range(n_locations))
                unvisited.remove(current_loc)
                path = [current_loc]
                distance = 0
                while unvisited:
                    next_loc = self.select_next_location(current_loc, unvisited, pheromone, distance_matrix)
                    path.append(next_loc)
                    distance += distance_matrix[current_loc][next_loc]
                    unvisited.remove(next_loc)
                    current_loc = next_loc
                distance += distance_matrix[path[-1]][path[0]]
                path.append(path[0])
                ant_paths.append(path)
                if distance < best_distance:
                    best_distance = distance
                    best_path = path
            for ant_path in ant_paths:
                for i in range(len(ant_path) - 1):
                    pheromone[ant_path[i]][ant_path[i+1]] += 1.0 / best_distance
                    pheromone[ant_path[i+1]][ant_path[i]] += 1.0 / best_distance
            pheromone *= (1 - self.decay)
            self.ant_paths_per_iter.append(ant_paths)
            self.best_path_per_iter.append(list(best_path))
            self.best_distance_per_iter.append(best_distance)
        return best_path, best_distance

def create_animation_frames(locations, aco, location_names):
    frames = []
    for iter_idx, (ant_paths, best_path) in enumerate(zip(aco.ant_paths_per_iter, aco.best_path_per_iter)):
        fig, ax = plt.subplots(figsize=(7, 5))
        x = [loc[0] for loc in locations]
        y = [loc[1] for loc in locations]
        ax.scatter(x, y, c='red', s=80, label='Locations')
        for i, name in enumerate(location_names):
            ax.annotate(name, (x[i], y[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)
        # Draw all ants
        for ant_path in ant_paths:
            ant_x = [locations[i][0] for i in ant_path]
            ant_y = [locations[i][1] for i in ant_path]
            ax.plot(ant_x, ant_y, 'o-', color='gray', alpha=0.2, linewidth=1)
        # Draw best path so far
        for i in range(len(best_path) - 1):
            loc1 = locations[best_path[i]]
            loc2 = locations[best_path[i + 1]]
            ax.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], 'b-', alpha=0.8, linewidth=2)
        ax.set_title(f'Iteration {iter_idx+1}')
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Longitude')
        ax.legend()
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert('RGB'))
    return frames

def frames_to_gif(frames, duration=100):
    buf = io.BytesIO()
    frames[0].save(buf, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)
    buf.seek(0)
    return buf

# --- Streamlit App ---
st.title('Ant Colony Optimization (ACO) for Delivery Route Planning')
st.write('Upload your delivery dataset and visualize the optimized route and animation!')

uploaded_file = st.file_uploader('Upload CSV', type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write('Data Preview:', df.head())
    columns = df.columns.tolist()
    lat_col = st.selectbox('Select latitude column', columns, index=columns.index('dest_lat') if 'dest_lat' in columns else 0)
    lng_col = st.selectbox('Select longitude column', columns, index=columns.index('dest_lng') if 'dest_lng' in columns else 1)
    locations_df = df[[lat_col, lng_col]].drop_duplicates()
    locations = list(zip(locations_df[lat_col], locations_df[lng_col]))
    location_names = [f"Location {i+1}" for i in range(len(locations))]
    st.sidebar.header('ACO Parameters')
    n_ants = st.sidebar.slider('Number of Ants', 5, 50, 15)
    n_iterations = st.sidebar.slider('Number of Iterations', 10, 100, 30)
    decay = st.sidebar.slider('Pheromone Decay', 0.01, 0.5, 0.1)
    alpha = st.sidebar.slider('Alpha (pheromone importance)', 0.1, 5.0, 1.0)
    beta = st.sidebar.slider('Beta (distance importance)', 0.1, 5.0, 2.0)
    if st.button('Run ACO'):
        aco = ACO(n_ants=n_ants, n_iterations=n_iterations, decay=decay, alpha=alpha, beta=beta)
        best_path, best_distance = aco.run(locations)
        st.success(f"Best path found: {best_path}")
        st.success(f"Total distance: {best_distance:.2f}")
        # Plot the best path
        fig, ax = plt.subplots(figsize=(10, 7))
        x = [locations[i][0] for i in range(len(locations))]
        y = [locations[i][1] for i in range(len(locations))]
        ax.scatter(x, y, c='red', s=100, label='Locations')
        for i, name in enumerate(location_names):
            ax.annotate(name, (x[i], y[i]), xytext=(5, 5), textcoords='offset points')
        for i in range(len(best_path) - 1):
            loc1 = locations[best_path[i]]
            loc2 = locations[best_path[i + 1]]
            ax.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], 'b-', alpha=0.7)
        ax.set_title('Best Delivery Route')
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Longitude')
        ax.legend()
        st.pyplot(fig)
        # Animation
        st.subheader('ACO Animation (Ants exploring and best path per iteration)')
        with st.spinner('Generating animation...'):
            frames = create_animation_frames(locations, aco, location_names)
            gif_buf = frames_to_gif(frames, duration=200)
            st.image(gif_buf, caption='ACO Animation', use_column_width=True)
else:
    st.info('Please upload a CSV file to begin.')
