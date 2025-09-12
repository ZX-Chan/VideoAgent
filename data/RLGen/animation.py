from manim import *

class RLGenAnimation(Scene):
    def construct(self):
        # Scene configuration
        self.camera.background_color = WHITE
        
        # ============ Component Creation Phase ============
        self.setup_components()
        
        # ============ Animation Sequence Execution ============
        self.play_all_scenes()
        
        # ============ End Processing ============
        self.wait(2)
    
    def setup_components(self):
        """Create all visual components"""
        self.components = {}
        
        # Generative Model
        self.components["comp_1"] = Rectangle(width=3, height=1.5, color=BLUE).shift(LEFT * 3 + UP * 2)
        gen_model_text = Text("Generative Model", font_size=24, color=BLACK).move_to(self.components["comp_1"].get_center())
        self.add(self.components["comp_1"], gen_model_text)
        
        # Agent
        self.components["comp_2"] = Circle(radius=1, color=GREEN).shift(UP * 2)
        agent_text = Text("Agent", font_size=24, color=BLACK).move_to(self.components["comp_2"].get_center())
        self.add(self.components["comp_2"], agent_text)
        
        # Environment
        self.components["comp_3"] = Rectangle(width=3, height=1.5, color=RED).shift(RIGHT * 3 + UP * 2)
        env_text = Text("Environment", font_size=24, color=BLACK).move_to(self.components["comp_3"].get_center())
        self.add(self.components["comp_3"], env_text)
        
        # Reward Signal
        self.components["comp_4"] = RegularPolygon(n=4, start_angle=PI/4, color=ORANGE).scale(1.2).shift(DOWN)
        reward_text = Text("Reward Signal", font_size=18, color=BLACK).move_to(self.components["comp_4"].get_center())
        self.add(self.components["comp_4"], reward_text)
        
        # Data Pool
        self.components["comp_5"] = Rectangle(width=2, height=1, color=GREEN).shift(DOWN * 2)
        data_pool_text = Text("Data Pool", font_size=18, color=BLACK).move_to(self.components["comp_5"].get_center())
        self.add(self.components["comp_5"], data_pool_text)
        
        # Annotations
        annotation = Text("RLGen Process", font_size=28, color=BLACK).shift(UP * 3.5)
        self.add(annotation)
        
        # Connections
        self.create_connections()
    
    def create_connections(self):
        """Create connections between components"""
        # Data flow from Generative Model to Agent
        arrow_1 = Arrow(self.components["comp_1"].get_right(), self.components["comp_2"].get_left(), buff=0.1, stroke_width=6)
        self.add(arrow_1)
        
        # Data flow from Agent to Environment
        arrow_2 = Arrow(self.components["comp_2"].get_right(), self.components["comp_3"].get_left(), buff=0.1, stroke_width=6)
        self.add(arrow_2)
        
        # Feedback from Environment to Reward Signal
        arrow_3 = CurvedArrow(self.components["comp_3"].get_bottom(), self.components["comp_4"].get_top(), angle=-PI/2, stroke_width=4)
        self.add(arrow_3)
        
        # Control flow from Reward Signal to Generative Model
        line_1 = DashedLine(self.components["comp_4"].get_left(), self.components["comp_1"].get_bottom(), dash_length=0.1)
        self.add(line_1)
        
        # Data flow from Environment to Data Pool
        arrow_4 = Arrow(self.components["comp_3"].get_bottom(), self.components["comp_5"].get_top(), buff=0.1, stroke_width=3)
        self.add(arrow_4)
    
    def play_all_scenes(self):
        """Execute all animation sequences"""
        # Scene 1: Framework Overview
        self.play(FadeIn(*self.components.values()), run_time=5, rate_func=smooth)
        # NARRATION: "This is the RLGen framework, combining generative models and reinforcement learning to mitigate data scarcity." (Duration: 5s)
        self.wait(5)
        
        # Scene 2: Data Generation
        self.play(GrowFromCenter(self.components["comp_1"]), run_time=5, rate_func=linear)
        self.wait(2)
        self.play(GrowFromCenter(self.components["comp_2"]), run_time=5, rate_func=linear)
        self.wait(2)
        self.play(Transform(self.components["comp_1"], self.components["comp_1"].copy().set_color(YELLOW)), run_time=5, rate_func=smooth)
        # NARRATION: "The generative model creates synthetic data, which the agent uses to generate tuples." (Duration: 5s)
        self.wait(5)
        
        # Scene 3: Tuple Evaluation
        self.play(GrowFromCenter(self.components["comp_3"]), run_time=5, rate_func=linear)
        self.wait(2)
        self.play(GrowFromCenter(self.components["comp_4"]), run_time=5, rate_func=linear)
        self.wait(2)
        self.play(Transform(self.components["comp_3"], self.components["comp_3"].copy().set_color(YELLOW)), run_time=5, rate_func=smooth)
        # NARRATION: "The environment evaluates the tuples, providing feedback to guide the generative model." (Duration: 5s)
        self.wait(5)
        
        # Scene 4: Data Storage
        self.play(GrowFromCenter(self.components["comp_5"]), run_time=5, rate_func=linear)
        # NARRATION: "The evaluated data is stored in a data pool for further analysis and use." (Duration: 5s)
        self.wait(5)