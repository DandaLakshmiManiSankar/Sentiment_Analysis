# Short Answer (Reasoning) [Part C]

**1. If you only had 200 labeled replies, how would you improve the model without collecting thousands more?**  
I would use transfer learning by starting with a pretrained DistilBERT model and fine-tuning it on the 200 replies with strong regularization and data augmentation (e.g., paraphrasing). Additionally, I could apply semi-supervised learning or few-shot techniques to leverage unlabeled email data for further improvements.

**2. How would you ensure your reply classifier doesn’t produce biased or unsafe outputs in production?**  
I would curate a diverse and representative training set and apply fairness checks across demographic groups. In production, I would monitor predictions continuously, add guardrails (e.g., filters for toxic outputs), and retrain the model periodically with feedback data.

**3. Suppose you want to generate personalized cold email openers using an LLM. What prompt design strategies would you use to keep outputs relevant and non-generic?**  
I would include structured context in the prompt, such as recipient’s role, company, and recent achievements, to ground the response. I’d also use few-shot examples of high-quality openers and add constraints like “be concise and specific” to avoid vague or generic outputs.
