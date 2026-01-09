---
layout: post
title: "Navigating the PhD Journey"
permalink: /blog/life-skills/navigating-phd-journey.html
---

**January 10, 2025** | *Life Skills & Career*

[← Back to Blog](../index.html)

---

## Introduction

Completing a PhD is often described as a marathon, not a sprint. After finishing my PhD in Computer Science at the University of Colorado Colorado Springs (2018-2022), I wanted to share some reflections and practical advice that I wish I had known at the beginning of my journey.

![UCCS](../Images/UCCS.png)

## The Early Days: Finding Your Footing

The first year of a PhD can be overwhelming. You're expected to take courses, explore research areas, find an advisor, and somehow figure out what you want to dedicate the next 4-5 years of your life to studying. Here's what helped me:

- **Read broadly, then narrow down:** Don't commit to a specific problem too quickly
- **Talk to senior PhD students:** They've been where you are and have valuable insights
- **Choose your advisor carefully:** This relationship will define your PhD experience
- **Build good habits early:** Time management and work-life balance from day one

> *"You do not need to know everything, everything you need to know you will figure it out when you need to know it."*

## The Middle Years: Deep in Research

Years 2-4 are when you do the bulk of your research. This is when imposter syndrome hits hardest, experiments fail repeatedly, and you wonder if you'll ever graduate. Some strategies that helped me stay productive:

### Managing Research

- **Keep detailed notes:** Future you will thank present you
- **Embrace failure:** Most experiments will fail, and that's okay
- **Version control everything:** Not just code, but ideas and papers too
- **Present regularly:** Lab meetings, conferences, even informal talks help clarify thinking

### Maintaining Mental Health

This cannot be overstated: your mental health is more important than any paper or deadline.

- Set boundaries between work and personal time
- Exercise regularly (even a 15-minute walk helps)
- Maintain friendships outside of academia
- Seek help when needed - there's no shame in therapy
- Remember why you started this journey

## The Final Push: Writing and Defending

The dissertation writing phase is unique. You're essentially writing a book while trying to wrap up final experiments and prepare for your defense. Tips that worked for me:

- **Start writing early:** Even rough drafts help organize your thoughts
- **Create a realistic timeline:** Writing always takes longer than you think
- **Get feedback frequently:** Don't wait until you have a "perfect" draft
- **Practice your defense talk:** Multiple times, with different audiences

## Lessons Learned

Looking back, here are the most valuable lessons from my PhD journey:

1. **It's a marathon, pace yourself:** Burnout is real and counterproductive
2. **Collaboration > Competition:** Your peers are not your rivals, they're your support network
3. **Done is better than perfect:** Especially for early projects and papers
4. **Learn to communicate:** Technical skills matter, but so does explaining your work clearly
5. **Build transferable skills:** Coding, writing, presenting - these serve you beyond academia
6. **Celebrate small wins:** Don't wait until graduation to acknowledge progress

## Code and Research Management

Here's a simple example of how I organized my research experiments:

```python
# experiment_tracker.py
import json
from datetime import datetime
from pathlib import Path

class ExperimentTracker:
    """Simple experiment tracking for research projects."""
    
    def __init__(self, log_dir="experiments"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
    
    def log_experiment(self, name, params, results):
        """Log experiment configuration and results."""
        experiment = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'parameters': params,
            'results': results
        }
        
        log_file = self.log_dir / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(log_file, 'w') as f:
            json.dump(experiment, f, indent=2)
        
        print(f"Logged experiment: {log_file}")
        return log_file

# Usage
tracker = ExperimentTracker()
tracker.log_experiment(
    name="objectosphere_cifar10",
    params={'learning_rate': 0.001, 'batch_size': 128, 'epochs': 100},
    results={'accuracy': 0.92, 'loss': 0.23}
)
```

This simple tracker saved me countless times when I needed to reproduce results or compare experiments!

## Life After the PhD

One question I get often: "Is a PhD worth it?" The answer depends on your goals. For me, it opened doors to interesting research problems and amazing colleagues. I now work as an Applied Scientist at [Samsara](https://www.samsara.com/), where I get to apply the research skills I developed during my PhD to real-world problems at scale.

## Final Thoughts

A PhD is challenging, but it's also an incredible opportunity for growth - both professionally and personally. You'll learn to tackle hard problems, think critically, and persist through uncertainty. These are valuable skills regardless of where your career takes you.

If you're considering a PhD or currently in one, feel free to reach out on [LinkedIn](https://www.linkedin.com/in/akshay-raj-dhamija/) or [Twitter](https://twitter.com/AkshayRDhamija). I'm always happy to chat and share experiences!

---

[← Back to Blog](../index.html)

