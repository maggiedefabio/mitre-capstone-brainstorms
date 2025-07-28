# mitre-capstone-brainstorms
model created during the 2025 mitre internship in cville, va.

Task: Learn from the LOCUST team about sUAS and come up with an OR problem that we
could develop and take into the academic year for our capstone project.

Our research question: Where should we launch our drones from, how should we allocate them, and how should we
schedule them so that they inflict maximum damage on the enemy while also maximizing strike
synchronization?

Scenario: Suppose we have 3 barges coming down a waterway and we are going to launch an
attack on up to 6 different targets. We do not have to strike all 6 targets, and our barges have 10
different launch points to choose from, though only one barge can launch per launch point. We
have 100 drones to allocate between the 6 sites and want to maximize damage. Each site requires a
certain number of successful drone hits to be fully destroyed, a weighted impact defining how highvalue they are, 
and a fortification score defining how well guarded they are. There is a baseline success of destroying each 
target, which increases when you allocate more drones and decreases when the launch point is further away. 
Additionally, our drones have a limited range, so not every target is a feasible option from every launch point.

