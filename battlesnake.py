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
def move(game_state: typing.Dict) -> typing.Dict:
    # start_time = time.perf_counter()
    is_move_safe = {
        "up": True,
        "down": True,
        "left": True,
        "right": True,
    }
    """
  This it the part to declare variables
  """
    length_of_body = game_state["you"]["length"]  # Length of your snake
    my_body = game_state["you"]["body"]
    my_head = game_state["you"]["head"]  # Coordinates of your head
    food = game_state["board"]["food"]  # Coordinates of food
    BOARD_WIDTH = game_state["board"]["width"]
    BOARD_HEIGHT = game_state["board"]["height"]
    health = game_state["you"]["health"]
    # opponents = game_state['board']['snakes']

    # coordinate of head
    head_coordinate_x = my_head["x"]
    head_coordinate_y = my_head["y"]

    # dictionary to convert direction to move to coordinate
    direction_to_move = {
        "up": (0, 1),
        "down": (0, -1),
        "right": (1, 0),
        "left": (-1, 0),
    }
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

    def is_next_move_food(next_move: Tuple[int, int]) -> bool:
        nonlocal food
        next_move_dict = {"x": next_move[0], "y": next_move[1]}
        return next_move_dict in food

    def future_body(
        body_now: List[Dict[str, int]],
        next_move: Tuple[int, int],
        eat_food: bool = False,
    ) -> List[Tuple[int, int]]:
        """Return the body in the future if you move to next_move"""
        body_next = body_now.copy()
        next_move_dict = {"x": next_move[0], "y": next_move[1]}
        # if not eat food, delete the last segment because the snake will move
        if not eat_food:
            body_next.pop()
        body_next.insert(0, next_move_dict)
        return body_next

    def distance_to_target(start: Dict[str, int], targets: Dict[str, int]) -> int:
        """Return the distance to the nearest food"""
        return abs(start["x"] - targets["x"]) + abs(start["y"] - targets["y"])

    class PathNode:
        """A node in the path"""

        def __init__(self, predict_steps, steps, body_now, eat_food=False):
            self.predict_steps = predict_steps
            self.steps = steps
            self.body_now = body_now
            self.eat_food = eat_food

        def __lt__(self, other):
            return int(self.predict_steps) < int(other.predict_steps)

    def how_many_step_to_target_by_A_star(
        body: List[Dict[str, int]],
        target: Dict[str, int],
        health_simulate: int = 100,  # for situation such as check if it is safe after eat food
        stay_away_from_food: bool = False,  # food will become roadblock if True
        find_a_longer_way: bool = False,  # if True, will find a longer way to target, which means it won't care about the visited coordinate
    ) -> List[Tuple[int, List[Dict[str, int]]]]:
        """
        Use A* to find the shortest path to target, will return shortest steps to the target. If the food is not reachable, the number of steps will be -1.

        Also, if it reach target, it will also store the coordinate of body in body_when_reach_target,
        store eat food in the path or not in eat_food_in_this_way.
        """
        head = body[0]
        path_to_target = []
        heapq.heapify(path_to_target)
        new_body_in_A_star = body.copy()
        nonlocal direction_to_move
        nonlocal BOARD_WIDTH
        nonlocal BOARD_HEIGHT
        visited = set()
        result_of_A_star = []

        if head in food:
            eat_food = True
        else:
            eat_food = False

        heapq.heappush(
            path_to_target,
            PathNode(
                1 + distance_to_target(head, target),  # f(n) = g(n) + h(n)
                1,
                body,
                eat_food,
            ),
        )

        while path_to_target:
            node = heapq.heappop(path_to_target)
            predict_steps, steps, body_now, eat_food = (
                node.predict_steps,
                node.steps,
                node.body_now,
                node.eat_food,
            )
            # if the snake will die before reaching the target, return -1
            if predict_steps > health_simulate:
                return result_of_A_star

            if stay_away_from_food and eat_food and body_now[0] != target:
                continue

            # if this step is the target, return the number of steps
            if body_now[0] == target:
                result_of_A_star.append((steps, body_now))
                if find_a_longer_way:
                    return result_of_A_star
                else:
                    continue

            on_a_food = True if body_now[0] in food else False

            # if visited, don't need to check it again
            if not find_a_longer_way:
                if (body_now[0]["x"], body_now[0]["y"]) in visited:
                    continue
                visited.add((body_now[0]["x"], body_now[0]["y"]))

            # from body in this situation, where can it go
            for index in direction_to_move.values():
                next_move = (body_now[0]["x"] + index[0], body_now[0]["y"] + index[1])
                next_move_eat_food = is_next_move_food(next_move) if not eat_food else True
                if not_to_run_into_wall(next_move):
                    # if the snake's head is on a food, means the snake will get longer in this turn
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

                    heapq.heappush(
                        path_to_target,
                        PathNode(
                            steps
                            + 1
                            + distance_to_target(
                                next_move_dict, target
                            ),  # f(n) = g(n) + h(n)
                            steps + 1,
                            new_body_in_A_star,
                            next_move_eat_food,
                        ),
                    )
        return result_of_A_star

    def check_and_select_move_to_tail(
        next_able_move: List[Dict[str, int]],
    ) -> typing.Dict:
        """When your health is greater than the number you set as NUM_OF_STEPS_TO_CHANGE_STRATEGY, chaise tail."""
        nonlocal able_to_move_but_not_safe
        nonlocal health
        nonlocal my_body

        able_to_move_and_safe_without_food_in_way = []
        able_to_move_and_safe_with_food_in_way = []

        for move in next_able_move:
            for direct, next_move in move.items():
                future_body_in_find_tail = (
                    future_body(my_body, next_move)
                    if health != 100
                    else future_body(my_body, next_move, True)
                )
                steps_to_tail_without_food = how_many_step_to_target_by_A_star(
                    future_body_in_find_tail,
                    my_body[-1],
                    health,
                    True,
                )
                steps_to_tail_with_food = how_many_step_to_target_by_A_star(
                    future_body_in_find_tail,
                    my_body[-1],
                    health,
                )

                if not steps_to_tail_without_food:
                    steps_to_tail_without_food = how_many_step_to_target_by_A_star(
                        future_body_in_find_tail,
                        my_body[-1],
                        health,
                        True,
                        True,
                    )
                if not steps_to_tail_with_food:
                    steps_to_tail_with_food = how_many_step_to_target_by_A_star(
                        future_body_in_find_tail,
                        my_body[-1],
                        health,
                        False,
                        True,
                    )

                if steps_to_tail_without_food:
                    heapq.heappush(
                        able_to_move_and_safe_without_food_in_way,
                        (steps_to_tail_without_food[0][0], direct),
                    )

                if steps_to_tail_with_food:
                    able_to_move_and_safe_with_food_in_way.append(
                        (steps_to_tail_with_food[0][0], direct)
                    )

        # print(f"Way without food: {able_to_move_and_safe_without_food_in_way}")
        # print(f"Way with food: {able_to_move_and_safe_with_food_in_way}")

        if able_to_move_and_safe_without_food_in_way:
            next_move = heapq.heappop(able_to_move_and_safe_without_food_in_way)[1]
            # print(f"MOVE {game_state['turn']:4}: {next_move:5}. Chaising tail! No food in the way!")
            return {"move": next_move}
        elif able_to_move_and_safe_with_food_in_way:
            next_move = heapq.heappop(able_to_move_and_safe_with_food_in_way)[1]
            # print(f"MOVE {game_state['turn']:4}: {next_move:5}. Chaising tail! Food in the way!")
            return {"move": next_move}

        return None

    def check_and_select_move_to_food(
        next_able_move: List[Dict[str, int]],
    ) -> typing.Dict:
        nonlocal health
        nonlocal my_body
        nonlocal food
        nonlocal able_to_move_but_not_safe

        safe_to_food_and_not_block_by_food_after = []
        safe_to_food_but_blocked_by_food_after = []

        for move in next_able_move:
            for direct, next_move in move.items():
                next_move_dict = {"x": next_move[0], "y": next_move[1]}
                there_is_a_food_in_this_way = (
                    False if not is_next_move_food(next_move) else True
                )
                future_body_in_find_food = future_body(my_body, next_move)
                food_need_to_search = (
                    [next_move_dict] if there_is_a_food_in_this_way else food
                )
                this_direction_is_safe = False
                for food_coordinate in food_need_to_search:
                    how_long_to_this_food_all = []

                    if there_is_a_food_in_this_way:
                        how_long_to_this_food_all = [(1, future_body_in_find_food)]
                    else:
                        how_long_to_this_food_all = how_many_step_to_target_by_A_star(
                            future_body_in_find_food, food_coordinate, health, True
                        )

                    if not how_long_to_this_food_all:
                        continue

                    for (
                        how_long_to_this_food,
                        body_when_eat_this_food,
                    ) in how_long_to_this_food_all:
                        this_food_is_safe = False
                        for move_after_eat_food in direction_to_move.values():
                            next_move_after_food = (
                                body_when_eat_this_food[0]["x"]
                                + move_after_eat_food[0],
                                body_when_eat_this_food[0]["y"]
                                + move_after_eat_food[1],
                            )
                            if not_to_run_into_wall(
                                next_move_after_food
                            ) and not_to_run_into_body(
                                body_when_eat_this_food, next_move_after_food, True
                            ):
                                new_tmp_body_to_find_tail_after_eat_food = future_body(
                                    body_when_eat_this_food,
                                    next_move_after_food,
                                    True,
                                )

                                after_eat_food_until_tail_without_food = (
                                    how_many_step_to_target_by_A_star(
                                        new_tmp_body_to_find_tail_after_eat_food,
                                        body_when_eat_this_food[-1],
                                        100,
                                        True,
                                    )
                                )
                                after_eat_food_until_tail_with_food = (
                                    how_many_step_to_target_by_A_star(
                                        new_tmp_body_to_find_tail_after_eat_food,
                                        body_when_eat_this_food[-1],
                                        100,
                                    )
                                )

                                if after_eat_food_until_tail_without_food:
                                    heapq.heappush(
                                        safe_to_food_and_not_block_by_food_after,
                                        (
                                            health - how_long_to_this_food,
                                            direct,
                                        ),
                                    )
                                if after_eat_food_until_tail_with_food:
                                    heapq.heappush(
                                        safe_to_food_but_blocked_by_food_after,
                                        (
                                            health - how_long_to_this_food,
                                            direct,
                                        ),
                                    )

                                if after_eat_food_until_tail_without_food:
                                    this_food_is_safe = True
                                    break
                                elif after_eat_food_until_tail_with_food:
                                    this_food_is_safe = True

                        if this_food_is_safe:
                            this_direction_is_safe = True
                            break

                if not this_direction_is_safe:
                    able_to_move_but_not_safe.append(direct)

        if safe_to_food_and_not_block_by_food_after:
            surplus, next_move_not_block_by_food_after = heapq.heappop(
                safe_to_food_and_not_block_by_food_after
            )
            # print(f"MOVE {game_state['turn']:4}: {next_move:5}. Chaising food! No food in the way after eating!")
            return (
                {"move": next_move_not_block_by_food_after}
                if surplus <= max(10, length_of_body - 3)
                else None
            )
        elif safe_to_food_but_blocked_by_food_after:
            surplus, next_move_blocked_by_food_after = heapq.heappop(
                safe_to_food_but_blocked_by_food_after
            )
            # print(f"MOVE {game_state['turn']:4}: {next_move:5}. Chaising food! Food in the way after eating!")
            return (
                {"move": next_move_blocked_by_food_after}
                if surplus <= max(10, length_of_body - 3)
                else None
            )
        else:
            return None

    next_able_move_is_food = []
    next_able_move_not_food = []
    next_able_move = []

    # check if the snake will run into itself or wall in the next turn
    for direct, index in direction_to_move.items():
        if is_move_safe[direct]:
            next_move = (head_coordinate_x + index[0], head_coordinate_y + index[1])
            health_is_100 = True if health == 100 else False
            if not not_to_run_into_body(
                my_body, next_move, health_is_100
            ) or not not_to_run_into_wall(next_move):
                is_move_safe[direct] = False
            else:
                if is_next_move_food(next_move):
                    next_able_move_is_food.append({direct: next_move})
                else:
                    next_able_move_not_food.append({direct: next_move})
                next_able_move.append({direct: next_move})

    if len(next_able_move) == 1:
        # print(f"MOVE {game_state['turn']:4}: {list(next_able_move[0].keys())[0]:5}. Only way to go!")
        # mid_time = time.perf_counter()
        # print(f"Time: {(mid_time - start_time)*1000:0.1f} ms")
        # print("--------------------------------------------------------")
        return {"move": list(next_able_move[0].keys())[0]}

    able_to_move_but_not_safe = []
    result = None
    no_safe_move_when_health_is_not_enough = False
    NUM_OF_STEPS_TO_CHANGE_STRATEGY = max(12, length_of_body)

    if (
        health > NUM_OF_STEPS_TO_CHANGE_STRATEGY and next_able_move
    ):  # if health is enough, check if the snake can reach tail in the future and be safe without eating food
        if next_able_move_not_food:
            result = check_and_select_move_to_tail(
                next_able_move_not_food,
            )
        if not result and next_able_move_is_food:
            result = check_and_select_move_to_tail(
                next_able_move_is_food,
            )
    elif (
        NUM_OF_STEPS_TO_CHANGE_STRATEGY >= health and next_able_move
    ):  # if health is not enough, check if the snake can reach food in the future and the (health) - (steps to food) is nearest to 0 but >= 0
        result = check_and_select_move_to_food(next_able_move)
        no_safe_move_when_health_is_not_enough = True if not result else False

    if no_safe_move_when_health_is_not_enough:
        if next_able_move_not_food:
            result = check_and_select_move_to_tail(
                next_able_move_not_food,
            )
        if not result and next_able_move_is_food:
            result = check_and_select_move_to_tail(
                next_able_move_is_food,
            )
    # end_time = time.perf_counter()

    if result:
        # print(f"Time: {(end_time - start_time)*1000:0.1f} ms")
        # print("--------------------------------------------------------")
        return result
    elif able_to_move_but_not_safe:
        # print(f"MOVE {game_state['turn']:4}: No safe moves detected! Moving random in {able_to_move_but_not_safe}")
        # print(f"Time: {(end_time - start_time)*1000:0.1f} ms")
        # print("--------------------------------------------------------")
        return {"move": random.choice(able_to_move_but_not_safe)}

    # print(f"MOVE {game_state['turn']:4}: No able moves detected! Moving down")
    # print("--------------------------------------------------------")
    return {"move": "down"}
# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
