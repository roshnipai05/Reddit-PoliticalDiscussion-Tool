# RAG System Evaluation Report

## 1. Configuration

- **Embedding model**: sentence-transformers/all-mpnet-base-v2
- **Vector store**: ChromaDB (collections: reddit_posts, reddit_comments)
- **Retrieval**: top-5 posts (diversity-filtered) + up to 5 comments per post
- **Re-ranking**: cosine similarity × log(1 + reddit_score)
- **No-answer threshold**: cosine similarity < 0.35
- **LLM A (Groq)**: llama-3.3-70b-versatile
- **LLM B (Gemini)**: gemini-2.0-flash  _(disabled — quota exhausted)_
- **Evaluation set**: 18 questions ({'factual': 9, 'opinion': 9})

---

## 2. Results Table

| ID | Type | Groq ROUGE-L | Gemini ROUGE-L | Groq BERTScore | Gemini BERTScore |
|----------------|----------------|----------------|----------------|----------------|----------------|
| q01 | factual | 0.0784 | 0.0000 | 0.6714 | 0.0000 |
| q02 | factual | 0.4231 | 0.0000 | 0.8850 | 0.0000 |
| q03 | factual | 0.0816 | 0.0000 | 0.6981 | 0.0000 |
| q04 | factual | 0.2270 | 0.0000 | 0.8467 | 0.0000 |
| q05 | factual | 0.1452 | 0.0000 | 0.7967 | 0.0000 |
| q06 | opinion | 0.0328 | 0.0000 | 0.6828 | 0.0000 |
| q07 | opinion | 0.1000 | 0.0000 | 0.7125 | 0.0000 |
| q08 | opinion | 0.1353 | 0.0000 | 0.7962 | 0.0000 |
| q09 | opinion | 0.0370 | 0.0000 | 0.7117 | 0.0000 |
| q10 | opinion | 0.0884 | 0.0000 | 0.7684 | 0.0000 |
| q11 | opinion | 0.0563 | 0.0000 | 0.6928 | 0.0000 |
| q12 | opinion | 0.0323 | 0.0000 | 0.7017 | 0.0000 |
| q13 | factual | 0.2817 | 0.0000 | 0.7821 | 0.0000 |
| q14 | factual | 0.0364 | 0.0000 | 0.6541 | 0.0000 |
| q15 | factual | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| q16 | factual | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| q17 | opinion | 0.1078 | 0.0000 | 0.7853 | 0.0000 |
| q18 | opinion | 0.1119 | 0.0000 | 0.7852 | 0.0000 |
| **Mean** | — | **0.2208** | **0.0000** | **0.7762** | **0.0000** |

---

## 3. Adversarial Question Behaviour

| ID | Question (truncated) | no_answer_flag | Groq Refused Correctly | Gemini Refused Correctly |
|---|---|---|---|---|
| q14 | What did users say about the results of the 2024 US Senate r... | ✗ | ✗ (check manually) | ✗ (check manually) |
| q15 | What is the most upvoted post of all time on this subreddit,... | ✗ | ✓ | ✗ (check manually) |
| q16 | How many subreddit moderators are there and what are their u... | ✗ | ✓ | ✗ (check manually) |

---

## 4. Performance by Question Type

| Type | Count | Groq ROUGE-L | Gemini ROUGE-L | Groq BERTScore | Gemini BERTScore |
|---|---|---|---|---|---|
| factual | 9 | 0.3637 | 0.0000 | 0.8149 | 0.0000 |
| opinion | 9 | 0.0780 | 0.0000 | 0.7374 | 0.0000 |

---

## 5. Full Model Answers

_Generated answers for manual review._

### q01 — factual 
**Question**: What was the most discussed political event in the subreddit during July 2024?
**Reference**: The two most discussed events in July 2024 were the Trump assassination attempt on July 13 and Biden's decision to exit the 2024 presidential race on July 21, both of which generated a significant volume of posts and comments.
**no_answer_flag**: False  |  **max_cosine_sim**: 0.6474

**Groq answer**:
> The corpus does not contain sufficient information to answer this question.

**Gemini answer**:
> _disabled_

---

### q02 — factual 
**Question**: What flair categories were most commonly used for posts about the 2024 presidential election?
**Reference**: Posts about the 2024 presidential election were most commonly flaired under categories related to US Elections, US Politics, and similar election-specific tags, reflecting the high volume of campaign and candidate-related discussion.
**no_answer_flag**: False  |  **max_cosine_sim**: 0.6861

**Groq answer**:
> The most commonly used flair categories for posts about the 2024 presidential election were US Elections and US Politics.

**Gemini answer**:
> _disabled_

---

### q03 — factual 
**Question**: How did users react to the Supreme Court's decision to overturn Chevron deference in June 2024?
**Reference**: Users expressed significant concern about the overturning of Chevron deference, with many commenters arguing it would weaken federal regulatory agencies and shift power to the courts. Critics framed it as a major setback for environmental and consumer protections.
**no_answer_flag**: False  |  **max_cosine_sim**: 0.7841

**Groq answer**:
> The corpus does not contain sufficient information to answer this question.

**Gemini answer**:
> _disabled_

---

### q04 — factual 
**Question**: Were there posts discussing third-party or independent candidates in the 2024 election?
**Reference**: Yes, there were posts discussing third-party candidates, particularly focusing on the viability and impact of independent or minor-party candidates on the 2024 election outcome, with debate over whether such candidates could affect the major-party results.
**no_answer_flag**: False  |  **max_cosine_sim**: 0.701

**Groq answer**:
> Reddit users discussed the possibility of third-party candidates in the context of future elections, such as 2028, but there were also posts inquiring about the 2024 election. Although the posts about the 2024 election did not specifically focus on third-party or independent candidates, users did express curiosity about the lack of independent party candidates and the possibility of a third-party president. Additionally, a post asked about all the 2024 running campaigns, which could imply interest in candidates beyond the two main parties. However, the discussion of third-party or independent candidates was more prominent in the context of future elections.

**Gemini answer**:
> _disabled_

---

### q05 — factual 
**Question**: What topics related to the economy were most discussed in the subreddit?
**Reference**: Economic discussions centred around inflation, housing costs, wages, and the overall cost of living, with users frequently debating the effectiveness of administration economic policy and comparing conditions under different political leadership.
**no_answer_flag**: False  |  **max_cosine_sim**: 0.6649

**Groq answer**:
> Reddit users discussed various topics related to the economy, including the impact of Trump-era policies on the current economy, the role of the economy in voting decisions, and general discussions about the economy in the context of US elections and politics. Some posts focused on the economy as a key talking point, while others explored specific aspects, such as the lingering effects of Trump-era policies. Overall, the economy was a significant topic of discussion, with users acknowledging its importance in shaping their voting decisions and the overall state of the country.

**Gemini answer**:
> _disabled_

---

### q06 — opinion 
**Question**: What do Reddit users in this community think about Biden's decision to withdraw from the 2024 presidential race?
**Reference**: Opinions were divided. Many users expressed relief or support, arguing that Biden stepping aside was the right move for the Democratic Party's chances against Trump. A minority expressed frustration or felt the decision came too late. A common sentiment was cautious optimism about Kamala Harris as the replacement candidate.
**no_answer_flag**: False  |  **max_cosine_sim**: 0.7757

**Groq answer**:
> The corpus does not contain sufficient information to answer this question.

**Gemini answer**:
> _disabled_

---

### q07 — opinion 
**Question**: What are the main arguments users make for and against Kamala Harris as a presidential candidate?
**Reference**: Arguments in favour of Harris included her prosecutorial background, her historic candidacy, and the view that she represented a meaningful contrast to Trump. Arguments against included concerns about her prior low polling numbers, questions about her policy clarity, and scepticism about whether she could consolidate enough swing-state support.
**no_answer_flag**: False  |  **max_cosine_sim**: 0.8634

**Groq answer**:
> The corpus does not contain sufficient information to answer this question.

**Gemini answer**:
> _disabled_

---

### q08 — opinion 
**Question**: How do users in this subreddit discuss the topic of immigration policy?
**Reference**: Immigration discussions were contentious. Some users advocated for stricter border enforcement and linked immigration to economic concerns, while others emphasised humanitarian considerations and criticised restrictive policies. A recurring theme was debate over the political consequences of immigration as an election issue.
**no_answer_flag**: False  |  **max_cosine_sim**: 0.6798

**Groq answer**:
> Users in this subreddit discuss the topic of immigration policy by presenting various perspectives and potential solutions. Some users argue that stricter border enforcement is necessary, while others advocate for pathways to citizenship. A few users suggest that the current system is flawed and that the US should adopt a more open approach, similar to what existed during the time of the founding fathers. 

Many users emphasize the need to address the root causes of undocumented immigration, such as companies and individuals who facilitate it by offering jobs and housing to undocumented workers. They propose increasing penalties for hiring undocumented labor to encourage businesses to reform their practices and support a more formalized immigration process.

Some users also highlight the historical context of US immigration policies, including the 1924 Immigration Act, which excluded certain groups based on race or nationality. Others note that anti-immigration views can be inconsistent and seemingly driven by various factors, including culture, religion, or politics.

Overall, the discussion on immigration policy in this subreddit is characterized by a range of opinions and ideas, with users engaging in a debate about the best approach to addressing the complex issues surrounding immigration. Reddit users tend to view immigration as a multifaceted issue that requires a comprehensive solution, taking into account both border security and the needs of immigrants and the US economy.

**Gemini answer**:
> _disabled_

---

### q09 — opinion 
**Question**: What is the general sentiment toward the Republican Party and Donald Trump in this community?
**Reference**: The general sentiment was predominantly critical. Users frequently expressed concern about Trump's policy positions, democratic norms, and character. However, some posts offered more neutral or analytical framings, and there were occasional users who defended Republican positions, particularly on economic and immigration policy.
**no_answer_flag**: False  |  **max_cosine_sim**: 0.7606

**Groq answer**:
> The corpus does not contain sufficient information to answer this question.

**Gemini answer**:
> _disabled_

---

### q10 — opinion 
**Question**: What do users think about the role of media coverage in the 2024 election?
**Reference**: Users were broadly critical of media coverage, with complaints spanning both mainstream and partisan outlets. Common criticisms included both-sidesism, sensationalism, and under-reporting of policy substance in favour of horserace coverage. Some users specifically criticised coverage of Biden's age while others argued Trump received insufficient scrutiny.
**no_answer_flag**: False  |  **max_cosine_sim**: 0.6996

**Groq answer**:
> Reddit users think that the media should fairly report on the 2024 election by providing accurate and unbiased information. They suggest that the press should report on what candidates actually say, verbatim, without interpreting or paraphrasing their statements. Users also believe that the media should not treat outright lies as valid arguments and should instead report on them as false. 

Some users are concerned about the role of social media in the election, with one user noting that hosts on social media are often entertainers rather than journalists, which can lead to a lack of scrutiny. Others think that social media will play an increasingly important role in electoral politics, with candidates pivoting towards online mediums over traditional TV-based mediums.

Users also emphasize the importance of transparency and accuracy in media coverage, suggesting that the media should show clips or transcribe quotes from candidates without editing or interpreting them. Additionally, users think that the media should not provide charitable characterizations of candidates' statements, but rather report on what they actually say.

Overall, Reddit users believe that the media has a significant impact on the election and that fair and accurate reporting is essential for informing voters and holding candidates accountable.

**Gemini answer**:
> _disabled_

---

### q11 — opinion 
**Question**: How did community discussion change after the Harris-Trump debate in September 2024?
**Reference**: Following the Harris-Trump debate in September 2024, the majority of posts reflected a view that Harris had performed strongly and Trump had underperformed. Users pointed to specific exchanges on abortion, democracy, and the economy as moments where Harris was seen to have the upper hand. Sentiment in the community appeared more optimistic about Harris's prospects after the debate.
**no_answer_flag**: False  |  **max_cosine_sim**: 0.7331

**Groq answer**:
> The corpus does not contain sufficient information to answer this question.

**Gemini answer**:
> _disabled_

---

### q12 — opinion 
**Question**: What do users say about voter suppression or election integrity concerns?
**Reference**: Users raised concerns about voter suppression through restrictive ID laws, polling access, and gerrymandering, primarily framing these as tactics used by Republican-controlled legislatures. There were also discussions about misinformation and the integrity of election administration, with most users expressing trust in the official processes while expressing concern about political interference.
**no_answer_flag**: False  |  **max_cosine_sim**: 0.7358

**Groq answer**:
> The corpus does not contain sufficient information to answer this question.

**Gemini answer**:
> _disabled_

---

### q13 — factual 
**Question**: Were there any posts discussing the Democratic National Convention in August 2024?
**Reference**: Yes, the Democratic National Convention in August 2024 was discussed in the subreddit, with posts covering speaker highlights, the official nomination of Kamala Harris, and debate about the convention's effectiveness as a campaign moment.
**no_answer_flag**: False  |  **max_cosine_sim**: 0.5959

**Groq answer**:
> There was a post discussing the 2024 Democratic National Convention Night 2 on August 20, 2024. The post is titled "2024 Democratic National Convention Night 2 8/20/24 Discussion" and has a score of 1.

**Gemini answer**:
> _disabled_

---

### q14 — factual (adversarial)
**Question**: What did users say about the results of the 2024 US Senate races and which party gained control?
**Reference**: The corpus does not contain sufficient information to answer this question.
**no_answer_flag**: False  |  **max_cosine_sim**: 0.6988

**Groq answer**:
> Reddit users discussed the 2024 US Senate races, with some mentioning that the Democrats would have to win all the states Biden carried in 2020, plus 2 seats that are currently controlled by Republicans. They noted that there was a path for Democrats through Ohio and Montana, but it would be hard. 

However, as the results came in, users expressed disappointment and concern as the Democrats seemed to be losing key seats. One user mentioned that losing Ohio was "terrible but not so shocking" and that if they lost other states like Michigan, Wisconsin, Pennsylvania, or Arizona, they wouldn't have a counter against the Republican agenda.

Another user stated that with Bernie Moreno being called in Ohio, it "pretty much guarantees the Republicans control of the Senate, potentially with an extra seat or two." This suggests that, according to the users, the Republican party gained control of the Senate in the 2024 elections.

**Gemini answer**:
> _disabled_

---

### q15 — factual (adversarial)
**Question**: What is the most upvoted post of all time on this subreddit, and what was it about?
**Reference**: The corpus does not contain sufficient information to answer this question.
**no_answer_flag**: False  |  **max_cosine_sim**: 0.4573

**Groq answer**:
> The corpus does not contain sufficient information to answer this question.

**Gemini answer**:
> _disabled_

---

### q16 — factual (adversarial)
**Question**: How many subreddit moderators are there and what are their usernames?
**Reference**: The corpus does not contain sufficient information to answer this question.
**no_answer_flag**: False  |  **max_cosine_sim**: 0.4515

**Groq answer**:
> The corpus does not contain sufficient information to answer this question.

**Gemini answer**:
> _disabled_

---

### q17 — opinion 
**Question**: What do users think about NATO expansion and its role in the Russia-Ukraine war?
**Reference**: Users discussing the Russia-Ukraine conflict generally expressed support for Ukraine and framed NATO as a defensive alliance rather than an aggressor. Criticism of Russia's invasion was dominant, with debate over the extent and nature of US military aid. Some users raised concerns about escalation risks while others argued for stronger support.
**no_answer_flag**: False  |  **max_cosine_sim**: 0.7967

**Groq answer**:
> Reddit users discuss the role of NATO in the Russia-Ukraine war, with some arguing that Russia's invasion of Ukraine was a result of NATO's expansion. However, others believe that Russia's actions are not justified and that NATO's expansion is not the primary cause of the conflict. 

One user suggests that if the US were to turn neutral, Ukraine's plan would be to acquire sufficient weapons, not rely on support in talks, as hugs with Putin won't help. Another user thinks that Ukraine should work to build nukes, as this seems to be the only thing that keeps a country from getting attacked.

Some users also discuss the US's role in NATO, with one user mentioning that the last defense spending bill included a provision that the US cannot leave NATO without congressional approval. However, another user notes that many scholars think this law may not hold up if brought before the Supreme Court.

In terms of NATO's expansion, users do not explicitly discuss the topic, but they do mention the strategic triangle of Russia, Iran, and China, and how these countries have imperialist aspirations. One user suggests that if only one of them made a move, NATO/Western countries would be able to support them, but a coordinated series of attacks would be more challenging to respond to.

Overall, Reddit users seem to think that NATO plays a crucial role in the Russia-Ukraine war, and that the US's involvement in NATO is important for maintaining stability in the region. However, there are differing opinions on the effectiveness of NATO's expansion and the US's ability to leave the alliance.

**Gemini answer**:
> _disabled_

---

### q18 — opinion 
**Question**: What do users say about the US healthcare system and potential policy reforms?
**Reference**: Healthcare was a recurring topic with users frequently criticising the cost and accessibility of the US system. Discussions often compared it unfavourably to universal systems in other countries. Policy debates centred on expanding Medicare, the Affordable Care Act, and drug pricing, with most users expressing support for more comprehensive public coverage.
**no_answer_flag**: False  |  **max_cosine_sim**: 0.7061

**Groq answer**:
> Reddit users express various concerns and opinions about the US healthcare system and potential policy reforms. Many users argue that the current system is flawed and prioritizes profits over people's health. Some commenters suggest that a single-payer system, such as "Medicare for All," could be a solution, while others propose a public option or a multiplayer system.

Several users criticize the idea of treating healthcare as a market-driven system, arguing that it fails to account for the needs of vulnerable populations. They also express skepticism about the ability of private insurance companies to provide adequate coverage, citing high costs and limited accessibility.

Some commenters propose specific policy changes, such as increasing residency slots for physicians, removing regulations that favor employer-based insurance plans, or implementing a public option with income-based premiums. Others emphasize the need to reduce the influence of corporate money and lobbyists in the healthcare system.

A few users mention the challenges of implementing universal healthcare, including the potential for a disastrous rollout, the impact on physician reimbursements, and the opposition from healthcare companies and lobbyists. Some commenters also note that many developed countries have achieved universal health insurance through multiplayer systems, which could be a more realistic and effective approach.

Overall, Reddit users express a range of opinions and concerns about the US healthcare system and potential policy reforms, highlighting the complexity and challenges of addressing this issue.

**Gemini answer**:
> _disabled_

---

## 6. Qualitative Analysis

_(Fill in after reviewing the full answers above.)_

### Where Groq succeeds

- 

### Where Groq fails

- 

### Where Gemini succeeds (once re-enabled)

- 

### Where Gemini fails

- 

### Retrieval quality observations

- 

### Adversarial behaviour

- 
