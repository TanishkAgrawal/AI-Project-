import streamlit as st
import joblib
import numpy as np

# Define the keyword list used in training
keywords = ["suicide", "suicidal", "kill myself", "want to die", "end my life", "death", "die", "dying", "hopeless", "depressed", "depression", "despair", "pain","parent",'drug',
    "suffering", "worthless", "no point", "can’t go on", "tired of living", "no reason to live", "give up", "want to end it", "feel alone", "lonely", "isolated", "empty","need someone",
    "numb", "self-harm", "cut", "overdose", "pills", "hang", "jump", "cliff", "gun",  "rope", "blade", "blood", "void", "dark", "black cloud", "no future", "can’t see a way out",'cry', 
    "life is pointless", "want peace", "escape", "burden", "failure", "useless","not good enough", "hate myself",  "self-hatred", "guilt", "shame", "regret", "no hope", "lost",'sick', 
    "broken", "can’t cope", "overwhelmed", "anxiety", "fear of living", "fear of death", "fear of failure",  "no one cares", "no one understands",  "no help",  "can’t get help",'kill',
    "therapist won’t help", "medication doesn’t work", "mental breakdown", "suicidal thoughts", "plan to die""method", "helium", "plastic bag", "drain cleaner", "fentanyl","i'm tired",
    "carbon monoxide", "poison","slit wrists", "crash car", "jump off bridge",  "no purpose", "life sucks", "misery", "agony", "torment", "trapped","stuck",  "can’t breathe","struggle",
    "chest hurts", "cry all the time", "scream inside","want to disappear", "better off dead", "world would be better without me", "family would be better off", "friends don’t care",
    "lost everyone", "abandoned", "betrayal", "trauma", "abuse", "physical pain", "emotional pain", "chronic pain", "can’t sleep", "sleepless nights", "nightmares", "dark thoughts",
    "violent thoughts", "obsession with death", "countdown to death", "last day", "final goodbye", "suicide note", "no one to talk to", "no support", "rejected", "unloved", 'thank','upset',
    "unlovable", "invisible", "ghost", "hollow", "soul crushed", "heart heavy", "stone in chest", "can’t feel joy", "can’t feel love", "life is a prison", "want to sleep forever", "broke",
    "never wake up", "eternal rest", "peace in death", "fear of waking up", "brain damage", "fail at suicide", "scared to fail", "scared to succeed", "guilt about family",'cant make friends',
    "hurt loved ones", "selfish to die", "can’t hurt others", "responsibility to live", "obligation to stay", "no escape", "endless struggle", "rock bottom", "can’t climb out",'feel trap',
    "bad day","im afraid","die","trigger","lost job", "relationship","cheat","kill myself","feel helpless", "can't take anymore", "i drank poison","depart this world" ,"fuck trippiersuicide",                  
    "utterly alone" ,"want fuck die","still hope gone","commit suicide day ago","don't know I'll kill in future","Antidepressants","Clinics", "Disappointment","cut","Low self-esteem",
    "hopeless", "despair" , "dysphoria", "helpless" , "lost","giving up","leave","hurt", "pain", "goodbye","Intrusive thoughts","Autopilot","Family issues","Anti-anxiety medication",
    "no hope","i'm a forgetting person", "Self murder" , "self destruction" , "chance - medley" , "Hari-kari", "homicide" , "man slaughter" , "foul play" , "slaying","Isolation","Purgatory",
    "Emotional numbness","life suck","brief depress","held little power" , "disconnected","lowest","distraught","drastic measure","unalive","unaliving","unalived","sewerslide","ending",
    "ghost","KMS","KYS","Menty B","Crash out","burnout","fade away","check out","Tap out","vanish","over it","endgame","exit stage left","no returns","don't want to live now", "drug overdosed"
    "spiral downward", "end the pain", "end it all", "meaningless","i can’t take it", "everything hurts", "take my life", "don’t want to live", "jump off", "tired of life", "done with life",
    "over it", "bleed out","hang myself", "overdose myself", "knife", "drown", "suffocate", "slit", "stab myself", "i wish i were dead", "nobody cares", "stress", "desperation", "anguish",
    "shattered", "falling apart", "nothing left", "want out", "too much", "breaking point", "hollow inside", "dead inside", "no strength", "exhausted", "crying", "sobbing", "panic","lone",
    "collapsing", "numbness", "bleeding", "razor", "shoot myself", "shoot", "gas", "toxic", "fatal", "goodbye", "last breath", "final moments", "done fighting", "let me go", "release me",
    "no way forward", "all over", "over", "finished", "wasting away", "wasting", "beyond help", "irreparable", "devastated", "therapy", "heroin", "overdose", "treat", "deserted", "hopeless",
    "embarrased", "embarass", "embarassing", "cheat", "cheated on", "ruin", "ruined", "dead", "coffin", "low life", "tragic", "overly sad", "past trauma", "pull trigger", "toxic relation",
    "suicide","beg","pathed","escape","fuck","alcohol"," self harm","empathyremors","selfconsci","relief","repress","TRI","diagnose","shotgun","disappear","antipsychotic","mind","guilt",
    "die","insult","illegitimate","shithead","stupidly","Ruin life", "scare","exhaust","goodbye","therapist","cry","dead hit", "dead","fear","hang","anguish","shoot","horrible","Fuckim",
    "jump","balcony","scream","silent","coward","scar","hate", "Gun","failure","depress","relationship","debt","girlfriend","anxious",  "psychiatrist","distraught","conscious","inferior",
    "dumb","physic","diminish", "overwhelm","angry","schizophrenia","miser","suicid","molest","harm","frustrate","divorce","fuck", "end","trauma","self-hate","jerk","lost","pointless","worthless",
    "struggle","rest","rest", "annoy","luckless","chance","peace","knife","bullshit","bleed","Jealousim","Nan", "schizophrenia","unhappy","ridicule","mistake","druggy","dumbass","commit",
    "escape","throwaway","pressure","hope","sync","careless","Subreddit","survive","assault", "painkillers","torture","burden","miser","serious","evil","distress","antidepress","Reddit",
    "suicide", "kill myself", "killing myself", "self harm", "depressed", "depression", "hopeless", "worthless", "give up", "end it all", "tired of life", "want to die", "can't go on",
    "no reason to live", "pain", "suffering", "die", "dying", "cut myself", "cutting", "hurt myself", "end my life", "hate myself", "life is pointless", "ending it", "I'm done", 'girlfriend',
    "take my life", "why live", "jump off", "pills", "overdose", "slit", "goodbye world", "empty", "lost", "alone", "lonely", "no one cares",  "anxiety", "crying", "hurting", "miserable",
    "despair", "overwhelmed", "helpless", "burden", "fail", "failure", "worthless", "can't sleep", "no hope", "trapped", "done", "numb", "exhausted", "can't take it", "can't handle",
    "hate life", "i'm not okay", "i give up", "nothing matters", "it's too much", "can't breathe", "sick of this", "hate everything", "i'm broken", "falling apart", "wish i could disappear",
    "don’t want to wake up", "no one understands", "not enough", "i failed", "everyone hates me", "suffocating", "wish I was gone", "just want peace", "can't keep going", "it hurts",
    "everything hurts", "so tired", "so alone", "can’t take the pain", "feel so lost", "can’t escape", "feel worthless","feel like a burden", "emptiness", "failure", "numb", "exhaustion",
    "despair", "overwhelmed", "broken", "worthlessness", "regret", "self-loathing", "void", "loneliness", "lost all hope", "no reason to continue", "everything feels pointless",'hard put'
    "wish I could freeze time", "life feels empty", "tired of pretending", "can’t stop crying", "I’m broken inside", "nothing makes sense anymore", "don’t want to wake up tomorrow",
    "everything hurts too much", "want to escape this suffering", "feel like I’m drowning", "want to disappear forever","hopelessness", "invisible", "broken", "anxious","I'm tired",
    "wish I was never born", "don’t see a way out", "I’m exhausted from life", "can’t take the pain anymore", "cant afford", "don't real", "goodbye", "sacrifice", "leave", "reach end",
    "Help", "drunk", "meth","methapthamine","jail","I'm done","sad feel","hurt","physic","hell", "problem","burn","burn alive","surgery","paralysis","unable","suffer","assualt","pill",
    "parental","horrible","heart break","blame","dysphoria","top floor","jump","one talk","nobody", "threat", "lost", "void","don't want", "fail","bull","weak","attempt","depersonalization",
    "unbearable", "I'm worst", "I'm burdern", "useless", "unless", "want talk", "sexual assualt", "dead","hang", "worry", "worried", "sorry", 'mommy', "daddy", "weird", "wrong", "ill",
    "freak", "creepy", "destroy", "wasted", "not left", "overdose", "divorce", "creep", "bastard", "trouble", "apology", "harass", "self harm", "shut", "restless", "lost mind", 'self cut',
    "lonely", "disappoint","disappointment","Im sorry", "asshol", "disappear", "ruin", "ruined", "hate", "disappear","loser", "evil", "coronavirus","hate", "Hate", "I'm exhaust", 
    "family probelm", "Im tire", "insecure","discomfort","unfair","don't want deal","homeless","lost hope", "garbage", "burden", "Im useless", "I'm killer", "I'm far gone", "suicide feelings",
    "end life", "don't want","anymore","disgust", "pointless", "quit", "sturggle", "worry","homesick", "End life",  "Cut", "shit", "ill jump", "can't bare", "exit", "note",  "I've never",
    "hate", "hate love","hate people", "stuck", "i'm sick", "I'm scare", "wish talk somebody", "scare", "anymore","hotline", "fail", "feel guilty", "last time", "break", "leaving this world",
    "anymore", "breakdowm", "isn't enough", "Cant take anymore", "I'm done", "I'm narcissist", "falut", "faluty", "panic attack", 'feel verbal', 'social inept', 'drug alcohol', 'alcohol',
    "I am scare", "nobody eve gave shit", "getting worse", " nobody help", "nobody care", "I am gonna be adult soon", 'wish I was 13 only', 'worthless', "can't do anything", "can't contribute",
    "don't want to give response", 'need attention', 'need care', 'dumb', 'psycho', 'panic attack', 'cant dress right way', 'cant do homework', "can't express", 'cant talk', 'cant fall in love',
    'dont want describe', 'breath hold', 'kill', 'random', 'yelling', 'finish', 'freak struggle', 'total thrash', 'self harm', "don't wanna exist", 'people want kill', 'freak', "don't like hear",
    'class drop', 'pretend', "didn't tell parents", 'drug', 'addiction', 'money', 'cheat', 'violent', 'hate', 'scream', 'escape', 'horrible', 'ignore', 'desire', 'dont care', 'anxious', 
    'cant meet new people', 'drop', 'TRI', 'i am stuck', 'no progress', 'depress', 'endless cycle', 'i am away',  'therapist ','i am bad explain people', "don't want to exist", 'go wrong', 
    'i dont care', 'lonely', 'hang', 'mental health', 'discord server', 'stress', 'burden', 'stupid' , 'hotline', 'rude', 'family problem', 'aggress', 'hung', 'offcut', 'really upset', 
    'uncertainity' , 'achieve goal', 'still alive', 'dont want live', 'shut', 'dont feel', 'fear', 'detach reality', 'lust', 'need advice', 'please', 'hard explain', 'disgust', 'i am sick',
    "don't fit", 'ask leave', 'feel like broken tool', 'nobody want around', 'life meaningless', 'danger', 'failure', 'refuse', ' biggest regret', 'fuck', 'suck', "don't like schol",
    'pain', 'breakdown', "didn't want to attend school", 'suicide', 'fail', "can't afford", 'failure', 'kill myself', 'house arrest', 'reckless drive', 'crash', 'self gun', 'Help please',
    'speechless', "couldn't even bare thought", 'please help',
  ]

# Load the vectorizer and model
try:
    vectorizer = joblib.load('vectorizer.pkl')
except Exception as e:
    st.error("Error loading vectorizer.pkl")
    st.stop()

try:
    model = joblib.load('model.pkl')
except Exception as e:
    st.error("Error loading model.pkl")
    st.stop()

# Set up Streamlit
st.set_page_config(page_title="Suicidal Tweet Detector", layout="centered", page_icon="🧠")

st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #ff4b4b;">🧠 Suicidal Tweet Detection</h1>
        <p style="font-size:18px;">Detect whether a tweet contains suicidal ideation using machine learning.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

tweet = st.text_area("✍️ Enter the tweet below:", height=150, placeholder="I feel like giving up...")

# Generate keyword_flag for this tweet
def get_keyword_flag(text):
    text_lower = text.lower()
    return int(any(kw in text_lower for kw in keywords))

# Predict
if st.button("🔍 Analyze Tweet"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet to analyze.")
    else:
        keyword_flag = get_keyword_flag(tweet)
        tweet_vector = vectorizer.transform([tweet])
        combined_vector = np.hstack([tweet_vector.toarray(), [[keyword_flag]]])
        prediction = model.predict(combined_vector)

        if prediction[0] == 1:
            st.success("✅ This tweet is **NOT suicidal**.")
        else:
            st.error("🚨 This tweet is **SUICIDAL**. Please seek support immediately.")
            st.markdown("""
                If you or someone you know is struggling, please contact a mental health professional or reach out to a support line in your country.
            """)

st.markdown("---")
st.markdown("<div style='text-align:center;'>Made with ❤️ for mental health awareness</div>", unsafe_allow_html=True)
