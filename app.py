import streamlit as st
import random
import time
from collections import defaultdict

# Grid settings
COLS = 9
ROWS = 7
ROAD_ROWS = [1,2,3,4,5]
FROG_START = (6,4)
NUM_CARS = 7

# RL parameters
ALPHA = 0.3
GAMMA = 0.9
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

REWARD_GOAL = 20
REWARD_HIT = -50
REWARD_STEP = -0.5

ACTIONS = ["forward","left","right"]
ACTION_DELTA = [(-1,0),(0,-1),(0,1)]

Q = defaultdict(lambda:[0,0,0])

class Car:
    def __init__(self,row):
        self.r=row
        self.c=random.randint(0,COLS-1)
        self.dir=random.choice([-1,1])

    def move(self):
        self.c += self.dir
        if self.c < 0:
            self.c = COLS-1
        if self.c >= COLS:
            self.c = 0

def random_cars():
    return [Car(ROAD_ROWS[i%len(ROAD_ROWS)]) for i in range(NUM_CARS)]

def state_key(frog,cars):
    grid=[[0]*COLS for _ in range(ROWS)]
    for car in cars:
        grid[car.r][car.c]=1
    grid_string="|".join("".join(map(str,row)) for row in grid)
    return f"{frog[0]},{frog[1]}|{grid_string}"

def collision(frog,cars):
    return any(frog[0]==c.r and frog[1]==c.c for c in cars)

def step(frog,cars):
    global EPSILON

    key=state_key(frog,cars)

    if random.random()<EPSILON:
        action=random.randint(0,2)
    else:
        action=Q[key].index(max(Q[key]))

    dr,dc=ACTION_DELTA[action]
    nr=max(0,min(ROWS-1,frog[0]+dr))
    nc=max(0,min(COLS-1,frog[1]+dc))
    newfrog=(nr,nc)

    for car in cars:
        car.move()

    reward=REWARD_STEP
    done=False

    if nr==0:
        reward=REWARD_GOAL
        done=True
    elif collision(newfrog,cars):
        reward=REWARD_HIT
        done=True

    new_key=state_key(newfrog,cars)
    old=Q[key][action]
    target=reward+(0 if done else GAMMA*max(Q[new_key]))
    Q[key][action]+=ALPHA*(target-old)

    return newfrog,reward,done

# ---------- UI ----------

st.title("🐸 Frogger RL (Live Training Demo)")

episodes = st.slider("Episodes", 10, 100, 30)
speed = st.slider("Speed (lower = faster)", 0.01, 0.3, 0.1)

grid_placeholder = st.empty()
info_placeholder = st.empty()

def draw_grid(frog, cars):
    grid = [["⬜"]*COLS for _ in range(ROWS)]

    for car in cars:
        grid[car.r][car.c] = "🚗"

    grid[frog[0]][frog[1]] = "🐸"

    return "\n".join([" ".join(row) for row in grid])

if st.button("Start Training"):
    frog = FROG_START
    cars = random_cars()

    success = 0

    for ep in range(episodes):
        done = False

        while not done:
            frog, reward, done = step(frog, cars)

            # Draw grid
            grid_placeholder.text(draw_grid(frog, cars))

            # Info
            info_placeholder.write(
                f"Episode: {ep+1}/{episodes} | Success: {success} | Epsilon: {round(EPSILON,2)}"
            )

            time.sleep(speed)

        if frog[0] == 0:
            success += 1

        frog = FROG_START
        cars = random_cars()

    st.success("Training Finished 🎉")
    st.write(f"Final Success Rate: {round(success/episodes*100,2)}%")