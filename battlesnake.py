import copy
import random
import typing
from typing import Dict, List, Tuple
import heapq
from collections import defaultdict, deque

# APIへのアクセスポイント
# Yamata-no-Orochi.pbl-b.repl.co

# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data


def info() -> typing.Dict:  # 蛇のカスタマイズ
    print("INFO")

    return {
        "apiversion": "1",
        "author": "g08",  # TODO: Your Battlesnake Username
        "color": "#ffd770",  # TODO: Choose color
        "head": "tongue",  # TODO: Choose head
        "tail": "weight",  # TODO: Choose tail
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
"""
This is a example of your data:
{
  "game": {
    "id": "totally-unique-game-id",
    "ruleset": {
      "name": "standard",
      "version": "v1.1.15",
      "settings": {
        "foodSpawnChance": 15,
        "minimumFood": 1,
        "hazardDamagePerTurn": 14
      }
    },
    "map": "standard",
    "source": "league",
    "timeout": 500
  },
  "turn": 14,
  "board": {
    "height": 11,
    "width": 11,
    "food": [
      {"x": 5, "y": 5}, 
      {"x": 9, "y": 0}, 
      {"x": 2, "y": 6}
    ],
    "hazards": [
      {"x": 3, "y": 2}
    ],
    "snakes": [
      {
        "id": "snake-508e96ac-94ad-11ea-bb37",
        "name": "My Snake",
        "health": 54,
        "body": [
          {"x": 0, "y": 0}, 
          {"x": 1, "y": 0}, 
          {"x": 2, "y": 0}
        ],
        "latency": "111",
        "head": {"x": 0, "y": 0},
        "length": 3,
        "shout": "why are we shouting??",
        "customizations":{
          "color":"#FF0000",
          "head":"pixel",
          "tail":"pixel"
        }
      }, 
      {
        "id": "snake-b67f4906-94ae-11ea-bb37",
        "name": "Another Snake",
        "health": 16,
        "body": [
          {"x": 5, "y": 4}, 
          {"x": 5, "y": 3}, 
          {"x": 6, "y": 3},
          {"x": 6, "y": 2}
        ],
        "latency": "222",
        "head": {"x": 5, "y": 4},
        "length": 4,
        "shout": "I'm not really sure...",
        "customizations":{
          "color":"#26CF04",
          "head":"silly",
          "tail":"curled"
        }
      }
    ]
  },
  "you": {
    "id": "snake-508e96ac-94ad-11ea-bb37",
    "name": "My Snake",
    "health": 54,
    "body": [
      {"x": 0, "y": 0}, 
      {"x": 1, "y": 0}, 
      {"x": 2, "y": 0}
    ],
    "latency": "111",
    "head": {"x": 0, "y": 0},
    "length": 3,
    "shout": "why are we shouting??",
    "customizations": {
      "color":"#FF0000",
      "head":"pixel",
      "tail":"pixel"
    }
  }
}
"""


def move(game_state: typing.Dict) -> typing.Dict:
    game_state["game"]["ruleset"]["settings"]["foodSpawnChance"] = 0
    game_state["game"]["ruleset"]["settings"]["minimumFood"] = 3
    is_move_safe = {
        "up": True,
        "down": True,
        "left": True,
        "right": True,
    }  # 移動可能な場所を初期化
    """
    This it the part to declare variables
    """
    length_of_body = game_state["you"]["length"]  # 自分の長さ
    my_body = game_state["you"]["body"]
    my_head = game_state["you"]["head"]  # Coordinates of your head
    food = game_state["board"]["food"]  # Coordinates of food
    BOARD_WIDTH = game_state["board"]["width"]
    BOARD_HEIGHT = game_state["board"]["height"]
    health = game_state["you"]["health"]
    # opponents = game_state['board']['snakes']

    head_coordinate_x = my_head["x"]
    head_coordinate_y = my_head["y"]

    body_when_reach_target = []  # will be append when running bfs

    direction_to_move = {
        "up": (0, 1),
        "down": (0, -1),
        "right": (1, 0),
        "left": (-1, 0),
    }  # dictionary to convert direction to move to coordinate
    """
    This is the part to put declare functions
    """

    # next_move is a tuple of (x, y)
    def not_to_run_into_wall(next_move: Tuple[int, int]) -> bool:
        """Check if the next_move is not in the wall"""
        nonlocal BOARD_WIDTH
        nonlocal BOARD_HEIGHT
        return (
            next_move[0] >= 0
            and next_move[0] < BOARD_WIDTH
            and next_move[1] >= 0
            and next_move[1] < BOARD_HEIGHT
        )

    # body_at_that_time is a list of dictionary of body coordinate, next_move is a tuple of (x, y), eat_food is a boolean
    def not_to_run_into_body(
        body_at_that_time: List[Dict[str, int]],
        next_move: Tuple[int, int],
        eat_food: bool = False,
    ) -> bool:
        """Check if the next_move is not in the body of the snake"""
        return (
            next_move
            not in [
                (segment["x"], segment["y"])
                for segment in body_at_that_time[: length_of_body - 1]
            ]
            if not eat_food
            else next_move
            not in [(segment["x"], segment["y"]) for segment in body_at_that_time]
        )

    # next_move is a tuple of (x, y)
    def is_next_move_food(next_move: Tuple[int, int]) -> bool:
        nonlocal food
        next_move_dict = {"x": next_move[0], "y": next_move[1]}
        return next_move_dict in food

    # body_now is a list of dictionary of body coordinate, next_move is a tuple of (x, y), eat_food is a boolean
    def future_body(
        body_now: List[Dict[str, int]],
        next_move: Tuple[int, int],
        eat_food: bool = False,
    ) -> List[Tuple[int, int]]:
        """Return the body in the future if you move to next_move"""
        body_next = body_now.copy()
        next_move_dict = {"x": next_move[0], "y": next_move[1]}
        if (
            not eat_food
        ):  # if not eat food, delete the last segment because the snake will move
            body_next.pop()
        body_next.insert(0, next_move_dict)
        return body_next

    # head is a dictionary of head coordinate
    def distance_to_target(head: Dict[str, int], targets: List[Dict[str, int]]) -> int:
        """Return the distance to the nearest food"""
        distance = []
        nonlocal food
        for target_coordinate in targets:
            distance.append(
                abs(head["x"] - target_coordinate["x"])
                + abs(head["y"] - target_coordinate["y"])
            )
        return min(distance)

    class PathNode:
        def __init__(self, predict_steps, steps, body_now):
            self.predict_steps = predict_steps
            self.steps = steps
            self.body_now = body_now

        def __lt__(self, other):
            return int(self.predict_steps) < int(other.predict_steps)

    # body is a list of dictionary of body coordinate, target is a dictionary of target coordinate
    def how_many_step_to_target_by_A_star(
        body: List[Dict[str, int]],
        target: Dict[str, int],
        health_simulate: int = 100,
        chaising_tail_after_eating: bool = False,
    ) -> List[int]:
        """
        Use A* to find the shortest path to food, will return a list of number of steps to each food. If the food is not reachable, the number of steps will be -1
        """
        head = body[0]
        path_to_target = []
        heapq.heapify(path_to_target)
        new_body_in_A_star = body.copy()
        nonlocal direction_to_move
        nonlocal BOARD_WIDTH
        nonlocal BOARD_HEIGHT
        nonlocal body_when_reach_target
        body_when_reach_target.clear()
        visited = set()

        heapq.heappush(
            path_to_target,
            PathNode(
                1 + distance_to_target(head, [target]),
                1,
                body,
            ),  # f(n) = g(n) + h(n)
        )

        while path_to_target:
            node = heapq.heappop(path_to_target)
            predict_steps, steps, body_now = (
                node.predict_steps,
                node.steps,
                node.body_now,
            )
            if (
                steps > health_simulate
            ):  # if the snake will die before reaching the target, return -1
                return -1

            if (
                body_now[0] == target
            ):  # if this step is the target, return the number of steps
                body_when_reach_target = copy.deepcopy(body_now)
                return steps

            tail_in_A_star = body_now[-1]
            on_a_food = (
                True
                if is_next_move_food((body_now[0]["x"], body_now[0]["y"]))
                else False
            )

            if (body_now[0]["x"], body_now[0]["y"]) in visited:  # if visited, continue
                continue
            visited.add((body_now[0]["x"], body_now[0]["y"]))

            for direct, index in direction_to_move.items():
                next_move = (body_now[0]["x"] + index[0], body_now[0]["y"] + index[1])
                if not_to_run_into_wall(next_move):
                    if on_a_food:
                        if not_to_run_into_body(body_now, next_move, True):
                            new_body_in_A_star = future_body(body_now, next_move, True)
                        else:
                            continue
                    else:
                        if not_to_run_into_body(body_now, next_move):
                            new_body_in_A_star = future_body(body_now, next_move)
                        else:
                            continue
                    next_move_dict = {"x": next_move[0], "y": next_move[1]}
                    if chaising_tail_after_eating:
                        target = tail_in_A_star

                    heapq.heappush(
                        path_to_target,
                        PathNode(
                            steps
                            + 1
                            + distance_to_target(
                                next_move_dict, [target]
                            ),  # f(n) = g(n) + h(n)
                            steps + 1,
                            new_body_in_A_star,
                        ),
                    )

        return -1

    # body is a list of dictionary of body coordinate, target is a dictionary of target coordinate
    def is_n_steps_after_safe_by_dfs(body: List[Dict[str, int]], n: int) -> bool:
        """
        Check if the snake is safe n steps after
        """
        path_after = deque()
        path_after.append((0, body))
        nonlocal direction_to_move
        nonlocal BOARD_WIDTH
        nonlocal BOARD_HEIGHT

        while path_after:
            steps, body_now = path_after.popleft()
            if steps == n:
                return True

            for direct, index in direction_to_move.items():
                next_move = (body_now[0]["x"] + index[0], body_now[0]["y"] + index[1])
                if not_to_run_into_wall(next_move) and not_to_run_into_body(
                    body, next_move
                ):
                    new_body_in_dfs = future_body(body_now, next_move)
                    path_after.appendleft((steps + 1, new_body_in_dfs))

        return False

    def check_and_select_move_to_tail(  # TODO
        next_able_move: List[Dict[str, int]],
        my_body: List[Dict[str, int]],
        able_to_move_and_safe: List[Tuple[int, str]],
    ) -> typing.Dict:
        nonlocal able_to_move_but_not_safe
        for move in next_able_move:
            for direct, next_move in move.items():
                steps_to_tail = how_many_step_to_target_by_A_star(
                    future_body(my_body, next_move), my_body[-1], health
                )
                if steps_to_tail == -1:
                    able_to_move_but_not_safe.append(direct)
                else:
                    heapq.heappush(able_to_move_and_safe, (steps_to_tail, direct))
        print(able_to_move_and_safe)
        if able_to_move_and_safe:
            next_move = heapq.heappop(able_to_move_and_safe)[1]
            print(f"MOVE {game_state['turn']:4}: {next_move:5}. Health is enough!")
            return {"move": next_move}

        return None

    def check_and_select_move_to_food(
        my_body: List[Dict[str, int]],
        next_able_move: List[Dict[str, int]],
        able_to_move_and_safe: List[Tuple[int, str]],
    ) -> typing.Dict:
        nonlocal health
        nonlocal body_when_reach_target
        nonlocal able_to_move_but_not_safe
        distance_each_way_to_each_food = [[] for _ in range(len(next_able_move))]

        for i, move in enumerate(next_able_move):
            for direct, next_move in move.items():
                future_body_in_find_food = future_body(my_body, next_move)
                for food_coordinate in food:
                    there_is_a_food_in_this_way = False
                    this_way_to_this_food = how_many_step_to_target_by_A_star(
                        future_body_in_find_food, food_coordinate, health
                    )
                    if this_way_to_this_food == -1:
                        continue
                    elif this_way_to_this_food == 1:
                        there_is_a_food_in_this_way = True

                    # check if the snake can be safe after eating the food (able to move safely to tail)
                    ok_to_eat = False
                    body_when_reach_target_in_find_food = copy.deepcopy(
                        body_when_reach_target
                    )
                    for move_in_direction in direction_to_move.values():
                        next_move = (
                            body_when_reach_target_in_find_food[0]["x"]
                            + move_in_direction[0],
                            body_when_reach_target_in_find_food[0]["y"]
                            + move_in_direction[1],
                        )
                        if not_to_run_into_wall(next_move) and not_to_run_into_body(
                            body_when_reach_target_in_find_food, next_move, True
                        ):
                            new_body_in_find_food = future_body(
                                body_when_reach_target_in_find_food, next_move, True
                            )
                            safe_after_eating = how_many_step_to_target_by_A_star(
                                new_body_in_find_food,
                                new_body_in_find_food[-1],
                                100,
                                True,
                            )
                            if safe_after_eating != -1:
                                ok_to_eat = True
                                break
                    print(direct, this_way_to_this_food, ok_to_eat)
                    if ok_to_eat:
                        distance_each_way_to_each_food[i].append(this_way_to_this_food)
                    elif there_is_a_food_in_this_way and not ok_to_eat:
                        distance_each_way_to_each_food[i].clear()
                        break

                if distance_each_way_to_each_food[i]:
                    print(distance_each_way_to_each_food[i])
                    segment_to_judge = (
                        max(distance_each_way_to_each_food[i])
                        if 1 not in distance_each_way_to_each_food[i]
                        else 1
                    )
                    heapq.heappush(
                        able_to_move_and_safe,
                        (health - segment_to_judge, direct),
                    )
                else:
                    able_to_move_but_not_safe.append(direct)
        print(able_to_move_and_safe)

        if able_to_move_and_safe:
            next_move = heapq.heappop(able_to_move_and_safe)[1]
            print(f"MOVE {game_state['turn']:4}: {next_move:5}. Health is not enough!")
            return {"move": next_move}

    next_able_move_is_food = []
    next_able_move_not_food = []
    next_able_move = []

    # check if the snake will run into itself or wall in the next turn
    for direct, index in direction_to_move.items():
        if is_move_safe[direct]:
            next_move = (head_coordinate_x + index[0], head_coordinate_y + index[1])
            if not not_to_run_into_body(my_body, next_move) or not not_to_run_into_wall(
                next_move
            ):
                is_move_safe[direct] = False
            else:
                if is_next_move_food(next_move):
                    next_able_move_is_food.append({direct: next_move})
                else:
                    next_able_move_not_food.append({direct: next_move})
                next_able_move.append({direct: next_move})

    if len(next_able_move) == 1:
        print(
            f"MOVE {game_state['turn']:4}: {list(next_able_move[0].keys())[0]:5}. Only way to go!"
        )
        return {"move": list(next_able_move[0].keys())[0]}

    able_to_move_but_not_safe = []
    able_to_move_and_safe = []
    result = None
    NUM_OF_STEPS_TO_CHANGE_STRATEGY = 15 + length_of_body // 3
    if (
        health > NUM_OF_STEPS_TO_CHANGE_STRATEGY and next_able_move
    ):  # if health is enough, check if the snake can reach tail in the future and be safe without eating food
        if next_able_move_not_food:
            result = check_and_select_move_to_tail(
                next_able_move_not_food,
                my_body,
                able_to_move_and_safe,
            )
        if not result and next_able_move_is_food:
            result = check_and_select_move_to_tail(
                next_able_move_is_food,
                my_body,
                able_to_move_and_safe,
            )
    elif (
        NUM_OF_STEPS_TO_CHANGE_STRATEGY >= health > 0 and next_able_move
    ):  # if health is not enough, check if the snake can reach food in the future and the (health) - (steps to food) is nearest to 0 but >= 0
        result = check_and_select_move_to_food(
            my_body, next_able_move, able_to_move_and_safe
        )

    if result:
        return result
    elif able_to_move_but_not_safe:
        print(
            f"MOVE {game_state['turn']:4}: No safe moves detected! Moving random in {able_to_move_but_not_safe}"
        )
        return {"move": random.choice(able_to_move_but_not_safe)}

    print(f"MOVE {game_state['turn']:4}: No safe moves detected! Moving down")
    return {"move": "down"}


# TODO: A*不一定用最好的姿勢去吃食物，有時候會有更好的姿勢


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
