# Prompt Engineering Guidelines

Prompts are instructions for large language models to generate a response or complete a task. The way you construct your prompt impacts the accuracy and format of the answer the model returns. That's why it's important to get your prompts right.

## General Tips

- Prompts can contain instructions or questions to pass to the model. They can also contain additional information, such as context or examples.
- Prompt engineering is an iterative process. Be ready to experiment.
- Version your prompts to compare the performance of different versions and choose the one that worked best.
- Start with simple, zero-shot prompts and then keep making them more complex to reach the required accuracy of answers. You can add more context and examples to each iteration of your prompt. If you're not seeing any improvements, fine-tune the model.
- Always use the latest models available.

## Prompt Format

Remember that the format of the prompt depends on the task you want the model to perform. Experiment and test what works best.

Prompts with the format that follows these guidelines seem to achieve better results than prompts with random formatting:

- The prompt has instructions, context, input data, and output format. Not all these components are necessary, and they may vary depending on the model's task.

You can try experimenting with specific prompt structures, for example:

```
Question:
Context:
Answer:
```

or for question answering without a given context:

```
Q:
A:
```

## Instructions

- Put them at the beginning of the prompt.
- Use commands, for example, "write", "summarize", "translate". Try using different keywords and see which yield the best results.
- Separate instructions from the context with a delimiter, for example, quotation marks (""") or hashes (##).
- Separate the instructions, examples, context, and input data with one or more line breaks.

### Examples:

```
## Instruction ##
Give a title to the article. The title must have 5 words.

Article: Online retail giant Amazon has said it plans to shutdown three warehouses in the UK putting 1200 jobs at risk.

Title:
```

```
Instructions: Detect the language of the text. Answer with the name of the language.

Text: Professionelle Beratung und ein Top-Service sind für uns selbstverständlich. Daher bringen unsere Mitarbeiter eine große Portion Kompetenz und Erfahrung mit.
```

## Context

Specify any information you want the model to use to generate the answer. The context should help the model arrive at better responses.

### Example:

```
Answer the question using the context. If you are not sure about the answer, answer with "I don't know".

Context: Contrails are a manmade type of cirrus cloud formed when water vapor from the exhaust of a jet engine condenses on particles, which come from either the surrounding air or the exhaust itself, and freezes, leaving behind a visible trail. The exhaust can also trigger the formation of cirrus by providing ice nuclei when there is an insufficient naturally-occurring supply in the atmosphere. One of the environmental impacts of aviation is that persistent contrails can form into large mats of cirrus, and increased air traffic has been implicated as one possible cause of the increasing frequency and amount of cirrus in Earth's atmosphere.

Question: Why do airplanes leave contrails in the sky?
```

## Output Format

Be specific about the format you want the model to use for the answer. If needed, specify the length, style, and so on. It may help to give the model a couple of concrete examples.

### Example:

```
Classify the text into neutral, negative, or positive.

Text: I love Berlin!
Answer:
```

By giving the model examples, you're showing exactly what answer you expect.

## Language

- Be direct, specific, and detailed.
- Be concise, as there's usually a limit on the number of tokens in a prompt and longer prompts are computationally more expensive than shorter ones.
- Focus on what you want the model to do rather than what you don't want it to do.

### Example:

```
Using specific language
Using imprecise language
```

```
Explain the concept nuclear sampling in 3 to 5 sentences to a 10-year old.
```

## Few-Shot Prompts

Few-shot prompts contain a couple of examples for the model. By giving the model examples, you demonstrate what you expect it to do and enable it to perform in-context learning to improve its performance.

- Use few-shot prompts if you require the model to return an answer in a specific format or if a simple prompt didn't yield the results you needed.
- Use consistent formatting.
- Provide examples in random order. For example, don't put all negative examples first and positive ones second. Mix them up as the order might bias the model.
- Make sure you use labels.

### Examples:

This example is from the Brown et al. 2020 paper:

```
A "whatpu" is a small, furry animal native to Tanzania. An example of a sentence that uses
the word whatpu is:
We were traveling in Africa and we saw these very cute whatpus.

To do a "farduddle" means to jump up and down really fast. An example of a sentence that uses
the word farduddle is:
Model answer:

One day when I was playing tag with my little sister, she got really excited and she
started doing these crazy farduddles.
```

Or for sentiment analysis:

```
Classify the text into neutral, negative, or positive.

Text: I love Berlin!
Answer: positive

Text: I hate Paris.
Answer: negative

Text: I've never been to Bangalore.
Answer: neutral

Text: I don't like Tokio.
Answer:
```