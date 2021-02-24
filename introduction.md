# Introduction

[Data science](https://en.wikipedia.org/wiki/Data_science) has become such a broad field that it is hard to provide a formal definition. The generic idea of associating patterns with tangible phenomena has been practiced for several millennia. For example, humans used the distinction between day and night, measured by the amount of sunlight, to adjust behaviors such as mobility, sleep, etc. - not because of fictional characters such as ghosts and demons, but because of predators and other tangible threats that put the life of one or more individuals or a population at risk. So, if you ever felt threated by the absence of sunlight and tried to stay away from it here's my response to you: "Congratulations! You just applied data science to reduce your total risk".

After reading my response one may ask the question - "Why is the risk higher at night?". To answer this question we to understand two terms: 1) risk, 2) causality. Risk, hazard and harm are different: if we treat harm as the measured outcome we can differentiate hazard and risk as follows: a hazard is a something that has the potential to cause harm (with non-zero probability), whereas risk is the quantification of the probability (some use quantity also, but I will avoid using it to be more principled) of harm.

Going out at night does not guarantee a harmful outcome, but there are few conditions where the risk is higher than usual. For example, leopards are nocturnal (active during the night) predators and are thought to be a threat to humans in places like Mumbai, India - read [this](https://www.theguardian.com/cities/2018/mar/23/mumbai-leopards-stray-dogs-protect-sanjay-gandhi-national-park) interesting article. We can easily infer that lack of sunlight is not the reason for the increase in probability of harm - lack of sunlight affects our vision, which exposes us to the true historical / potential **causes** (superset of hazards in this case) of harm.

Ghosts and demons are unreliable risk factors because: a) detection of a leopard is objective and does not vary across observers whereas detection of ghosts is highly variable within and between observers due to biases, b) even if ghosts are real and if a set of rules are estabilished for the detection of ghosts (independent of observers), evidence should be provided to prove the increase in risk due to ghosts either through controlled or observational studies. Thinking that ghosts are a threat to humans without checking these two criteria is an emotional decision.

Emotions play a significant role in every day decisions - for example, people believe that travel by air is riskier than travel by road - read [this](https://traveltips.usatoday.com/air-travel-safer-car-travel-1581.html) article for a summary. In the remainder of this book we make a clear distinction - luck, ghosts, and emotions are not scientific and will not be a part of causal explanations given to observed phenomena. The 'effect' of these factors will be quantified if possible - this activity is called uncertainty quantification.

<Write about the types of uncertainties: uncertainty in outcome, uncertainty in predictors for the given example>

## The data scientist in scientists

The first *formal* use of data science methods was done by [Sir Francis Galton](https://en.wikipedia.org/wiki/Francis_Galton) - reading only the third paragraph will give goosebumps. He observed that extreme characteristics of parents such as height were not passed on completely to the offspring - a concept called [regression to the mean](https://en.wikipedia.org/wiki/Regression_toward_the_mean). In simpler terms, if we assume the parents are the first generation and the offsprings are the second generation, an offspring is expected to be fewer deviations away (with respect to the second generation) from the mean compared to the parent (with respect to the first generation). In chapter <1> we will reproduce his analysis by analyzing the same data set.

Galton, despite his brilliance, believed in eugenics because the data provided evidence in its favor. However, in modern days eugenics is considered as an [unethical practice](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1129063/). But the stage was set for data science - methods for doing data science such as correlation and regression analysis became popular. Statistical measures such as **mean** became common in lab experiments - for example, in lab experiments to:

1. *Estimate* the value of acceleration due to gravity using a pendulum and a digital clock
2. *Estimate* the focal length of a convex lens by focusing a long-distant object on a screen
3. *Estimate* the tension of a string using a wedge and a tuning fork

In each case the experiment was repeated several times and the average value was calculated to estimate a physical quantity. Despite all the calculations I had no idea that averaging was done to reduce (standard) error. When I saw the similarities across experiments I understood the link - measurements vary across experiments because of a) uncontrollable factors, 2) measuring instrument, 3) observer or individual who is recording the measurement. It finally became clear to me that statistics is essential to do science. When I started looking at scientists with this filter it became clear to me that all scientists are trying to fit models to explain the data. Voila!

## Re-examining Newton's law of gravitation

### Short story inspired by [Cosmos: A Spacetime Odyssey](https://en.wikipedia.org/wiki/Cosmos:_A_Spacetime_Odyssey)

[Sir Isaac Newton](https://en.wikipedia.org/wiki/Isaac_Newton) has contributed to several domains - astronomy, mathematics, and theology to name a few. His universal law of gravitation was one of greatest triumphs in astrophysics that allowed scientists to use a simple law that is applicable on Earth to astronomical objects that were far beyond our reach. [Edmund Halley](https://en.wikipedia.org/wiki/Edmond_Halley) used the law to estimate that a comet that appeared in 1682 was identical to two comets that appeared in 1531 and 1607. Using just the law he predicted that the comet will reappear in 1758. Unlike religious predictions of apocalyptic end of life on Earth, the stakes were extremely high for this prediction. As predicted by Halley the comet appeared in 1758, which he did not live to see.

### Newton's law of gravitation as data analysis

Newton did his part in the formulation of the law of gravitation. It was the most prolific scientific achievement of the century that could not be accomplished by other greats like [Robert Hooke](https://en.wikipedia.org/wiki/Robert_Hooke), the inventor of [Hooke's law](https://en.wikipedia.org/wiki/Hooke%27s_law) who is also known for coining the term 'cell'. Creation of the law was facilitated by the contributions of [Tycho Brahe](https://en.wikipedia.org/wiki/Tycho_Brahe) towards data collection and [Johannes Kepler](https://en.wikipedia.org/wiki/Johannes_Kepler) towards planetary motion and heliocentrism.

My personal curiosity was sparked by one question: how do all the observations bind together to tell a coherent story, so I read the [Principia Mathematica](https://en.wikipedia.org/wiki/Principia_Mathematica). Finally it made complete sense - the book used observations from Earth to systematically argue why Mercury and Venus were inferior planets and the other three (Uranus (1781) and Neptune (1846) were discovered later) were superior planets, why Kepler used elliptical orbits for planets (visualizations for inferring elliptical orbits) and extensions of Kepler's laws (estimation of area swept in equal intervals of time) to form an elegant equation.

Today we are aware that Newton's universal law of gravitation is not accurate enough to explain the motion of Mercury. It took another scientific genius ([Albert Einstein](https://en.wikipedia.org/wiki/Albert_Einstein)) to fit the curve in a better way and simultaneously come up with a law that generalizes well to astronomical objects that were not observed during his lifetime. This is (data) science at work!

## Theory vs practice

### Examples of experimentation driving science

With all due respect to experimentalists I'm presenting my limited understanding of the role of experimentation in science before presenting my understanding of the role of theory. It's not hard to guess that I strongly favor the *need* for theory and first principles to drive science. However, historically this has not been the case and experimentalists have made giant leaps in the direction of scientific progress. There are several examples of such advancements; let's focus on the work of Ernest Rutherford.

["If your experiment needs statistics, you ought to have done a better experiment"](https://www.quora.com/What-did-Ernest-Rutherford-mean-by-If-your-experiment-needs-statistics-you-ought-to-have-done-a-better-experiment) - Ernest Rutherford

### Examples of theory driving science

Section to be filled

### Some thoughts on theory vs practice

Rutherford's picture is not complete. It's often not easy to experiment with a system by varying one measurement while controlling all other measurements. For example, in order to study the effect of 1918 influenza pandemic on human immunity it's not possible to travel back in time to change the progression of the disease to observe the effect on today's average immunity to flu antigens. It may be appropriate to say "Ok, instead I will clone an identical antigen and allow it to spread in 2020", but the average human immunity to influenza has changed between 1918 and 2020. Non-stationarity of the outcome makes experimentation extremely hard. In case of non-stationarity the best solution is to perform simultaneous test-control studies. [This](https://physicstoday.scitation.org/do/10.1063/PT.5.010218/full/) article provides insights on the need for *correct* models to test hypotheses. However, it does not dismiss the **need** for experimentation. Currently experimentation is the best way to establish **causality**. Statistical methods for establishing causality are evolving over the years, but are relatively immature.

## Personal Stories

Disclaimer: I consider this section a 'fall from grace' because we are transitioning instantaneously from incredible science (repeatability, reproducibility, predictive power, etc.) to a highly subjective topic: emotions.

This content has been moved to [Stories of a (Data) Scientist](https://snaveenmathew.github.io/randomly_accessed_memories/data_scientist.html) section of [Randomly Accessed Memories](https://snaveenmathew.github.io/randomly_accessed_memories)

## Tying the knots together: guidelines for data scientists

The introduction started with tall claims of science, discussed the need for a mix of theory and practice, and discussed how the industry is actually structured. I'd like to end this section on a positive note with some of my own guidelines for a career in data science. These guidelines are tailor-made for people who are pedantic and are not meant to be exhaustive.

### Guideline 1: import sklearn

Yes, you read that right! Please learn to `import sklearn`. Why? Think of this analogy: even today people who study pure mathematics are considered [unemployable in industry](https://www.quora.com/What-are-common-jobs-for-people-with-pure-mathematics-degrees-e-g-a-B-S-in-Mathematics) because of the enigma that surrounds mathematicians. They are thought by many as people who live in an ideal world that doesn't exist and that their "models" are purely theoretical and have no relevance to the real world. Even though the perception has changed over time, there is some truth to this statement.

The corporate world considers only two financial concepts: cost and return-on-investments (ROI). While it's fair to consider a company as a "going concern" and to perform valuation using an infinite sum ([discounted cash flow](https://www.investopedia.com/terms/d/dcf.asp)), it is important to have a continuous source of revenue (short-term) before involving in moonshots (long-term). As a results, most small/medium sized companies are evaluated purely on short-term returns. This view may not be wrong because moonshots require significant amount of time and investment. However, this significantly affects the portfolio. Companies tend to invest in the choices with least risk. This act tends to produce thousands of identical small / medium sized companies with different names that compete on price charged on the client (business / individual). I'm a product of one of those companies.

I'm not going to comment on a way to create a company with a balanced portfolio of mundane projects and mooonshots because the decision makers in those thousand identical companies are wiser than me. I'm going to share a way to work as an ethical data scientist. A good data scientist identifies KPIs that count and identifies the levers that affect the KPI. Therefore, a good data scientist is expected to know how to evaluate oneself. Since the company is more likely to care about cost and ROI, a good data scientist should be able to: a) fetch more revenue than the cost-to-company (CTC, often larger than gross salary), b) show higher ROI as a justification for growth within the company.

In order to achieve short-term goals it is essential to build generic prototypes using predefined functions. Therefore, it becomes critical to use packages like `sklearn` and `caret` to demonstrate the potential returns. This may differ from the long term goals of the individual (like me, who wants to understand the details from theory to practice), but is a company's unexpressed *need*. There's no point in achieving greatness in machine learning without fulfilling this need.

To summarize: do whatever everyone else is doing. However, as one might imagine, this is not enough. One needs to do more, which takes us to guideline 2.

### Guideline 2: learn and understand classical methods before going 'deep'

Going through classical methods is like going through the history of a field. For example, natural language processing started humbly by assuming 'he' and 'him' as unrelated words. Today's methods perform some for of magic (more on this later) to learn representations that relate 'he' and 'him'. The transition is smoother than one may imagine. This makes it mandatory to understand that methods that were on the way between [DTM](https://en.wikipedia.org/wiki/Document-term_matrix) and [GPT-3](https://en.wikipedia.org/wiki/GPT-3). Marketeting GPT-3 as the ultimate AI (as most people do) is not going to help. But, as mentioned in point #1, it is essential to have a working implementation of GPT-3 to showcase one's employability.

### Guideline 3: never stop learning

I don't have to explain this in detail. The landscape of machine learning is evolving rapidly. Research papers like [this](https://arxiv.org/pdf/1611.03530) have pointed out the need for rigorous theoretical frameworks to redesign the way we understand machine learning. Some of these views could be wrong, but it is essential to scrutinize all views in a scientific and logical way. This requires tremendous amount of effort because tremendously large number of streams show improvements either theoretically or empirically. Also, rationalizing improvements across several streams requires solid theoretical foundations.

This is also closely related to ROI and guideline #1: try new things. If nearly 90% of the 'deep learning specialists' built an `opencv` (sigh!) application for face detection from live webcam feed, you should have one too!

### Guideline 4: build your portfolio

The last sentence transitions smoothly into the next guideline: your portfolio defines your ability to produce high ROI. There are several ways to build a portfolio:

1. Kaggle (irony!): this is considered as the de facto standard
2. GitHub: build your own code and open-source, or copy code (with acknowledgements) and build on it
3. Contributions: work on widening, optimizing or improving existing packages like `sklearn`. This may be challenging because thousands of people work together to produce optimized code. Alternatively, one can identify a need and work on an open-source package that caters to the need

### Bonus (not mandatory): read research, try 'papers with code'

As mentioned in the first paragraph of guideline #3, it may be hard to have a consistent understanding of research because of the diverse nature of recent publications. This is especially true with publications that use biased data sets that give empirically better results. This makes the comparison of two empirical results hard. The best approach to solve this problem is to download the code and to examine their experimental behavior on different data sets. [Papers with code](https://paperswithcode.com/) is an excellent starting point. This step is essential to characterize the behavior of models against different types of data sets, especially in areas where ['transfer learning'](https://en.wikipedia.org/wiki/Transfer_learning) feature extractors are not mature.

### Personal guideline: first principles!

![](data/ML_leap_of_faith.jpg)

Forbes predicted that [in the United States there will be another 250,000 data scientist job openings by 2024](https://www.forbes.com/sites/joewalleneurope/2019/03/26/can-outsourcing-data-science-fill-the-jobs-shortage-fayrix-believes-so/#54eda092bce7). Given the current shortage the gap between demand and supply of data scientists is expetcted to grown. In fact the situation has changed to such an extent that (data science) training institutes are pointing fingers at academia. In some cases data science experts (sigh!) are encouraging students to directly enrol in training institutes instead of applying for a bachelors degree. What does it really take to become a data scientist?

Most machine learning experts (sigh!), trainers and self-proclaimed gurus claim that domain knowledge, SQL and scikit-learn are sufficient to make a complete data scientist. If every new data science candidate follows this approach in the United States we will have an increase in demand-supply gap of 250,000 by 2024.

The data science industry is moving towards black-box models. Sales people, data science experts, trainers and self-proclaimed gurus get a high when they say "We built a complicated deep <insert your architecture here> model that gave xx.xx% accuracy". The true interpretation of this statement is - "I copied some code from GitHub and struggled hard to get it to work for my data set. Finally I identified and tuned hyperparameters" without actually knowing why <insert your architecture here> works.

Sometimes these experts deliver a model that takes x days to train. Then identify inefficiencies in their code, optimize the code to run in x/1000 days, and claim that they saved \\$x million for their client by optimizing the code. This is unethical! The truth is - the client spent \\$y million more because of the inefficient code that was submitted earlier. Why didn't the data scientist identify inefficiencies in the code before the first release?

The answer is surprisingly simple - the experts, trainers and gurus who advise students to "think beyond bachelors degree" (Indians who read this will recall a similar statement by a famous person) and prepare cheatsheets for trivial problems failed to teach first principles. Suboptimal code is produced when a graduate of the data science training academy faces a problem that's not in the cheatsheet. A handful of graduates eventually understand first principles by reading zillions of Stack Overflow / Stack Exchange issues, which helps them in solving inefficiencies.

Training institutes are like factories - their outputs are almost identical. Sometimes academia participates in mass production - especially when the fee is high and ROI is low, where students who take risks are systematically disassociated and removed from 'statistics' (reports). This ecosystem is sustained by companies that need people who roll-over on command without thinking. I was the product of such an ecosystem. I joined factories not once, not twice, but thrice (as of September 2020).

After some introspection at the turn of the decade (2019-2020) I came to the following conclusions:

1. Around 2014 I made a skillset map with 3 colors: red (no/little understanding of theory and no/little practice), orange (no/little understanding of theory and medium/high practice), green (medium/high understanding of theory and medium/high practice). Several areas that were red/orange in 2014 became green in early 2020
2. It took few drastic changes to bring my career back on track
3. It took me **13+ years** of semi-structured learning to gain a half-decent understanding of machine learning

Several people use high school mathematics only for posting "1 + 1 - 1 * 1 + 1 / 1 = ?" on social media. Several people claim that high school mathematics was totally useless because it did not help them in doing their taxes. Data science is not such a career. Strong foundations in mathematics and computation are required for a career in data science.

## Objective of the 'book'

"Theoretically sound practice" of machine learning is a common area that's explored by everyone. It has become so common that memes such as "University professor vs that Indian guy on YouTube" get liked by majority of the people. This book attempts to provide "practically sound theory". The motivation is to fill the every-growing gap between theory and practice. One may assume that "theoretically sound practice" is sufficient to bridge the gap between theory and practice, but this is clearly not the case.

As stated in the preface and earlier sections of the introduction, this book is meant for people who are interested in details. The preliminary structure (2020/08/31) is as follows:

1. Exploratory data analysis - building plots from scratch to building plots using `matplotlib` / `ggplot2` and beyond
2. Basic statistics - mathematical foundations of point estimation, bias-variance tradeoff, hypothesis testing and inference
3. Statistical models - definition, parameter estimation, motivation for different optimization methods and inference
4. Machine learning - similar to statistical models
5. Deep learning - similar to statistical models
6. Efficiency (storage, retrieval, memory, computation, etc.)
