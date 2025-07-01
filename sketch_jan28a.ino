#define GREEN1_PIN 5
#define RED1_PIN 4
#define GREEN2_PIN 15
#define RED2_PIN 19
#define Yellow1_pin 23
#define Yellow2_pin 21

char currentCommand = '\0'; // Current command being executed
char nextCommand = '\0';    // Next command to be executed after transition
unsigned long phaseStart = 0;
uint8_t currentPhase = 0;
bool isTransitioning = false; // Flag to indicate if we're in a global transition phase

void allOff() {
  digitalWrite(GREEN1_PIN, LOW);
  digitalWrite(RED1_PIN, LOW);
  digitalWrite(GREEN2_PIN, LOW);
  digitalWrite(RED2_PIN, LOW);
  digitalWrite(Yellow1_pin, LOW);
  digitalWrite(Yellow2_pin, LOW);
}

void setup() {
  Serial.begin(115200);
  pinMode(GREEN1_PIN, OUTPUT);
  pinMode(RED1_PIN, OUTPUT);
  pinMode(GREEN2_PIN, OUTPUT);
  pinMode(RED2_PIN, OUTPUT);
  pinMode(Yellow1_pin, OUTPUT);
  pinMode(Yellow2_pin, OUTPUT);
  allOff();
}

void handleCommand(char cmd) {
  if (currentCommand == '\0') {
    // No current command, start immediately
    currentCommand = cmd;
    phaseStart = millis();
    allOff();
  } else {
    // Store the next command and start the transition phase
    nextCommand = cmd;
    isTransitioning = true;
    phaseStart = millis();
    allOff();
    digitalWrite(Yellow1_pin, HIGH);
    digitalWrite(Yellow2_pin, HIGH);
  }
}

void processPhase() {
  if (currentCommand == '\0') return;

  unsigned long now = millis();
  unsigned long phaseDuration = 0;

  if (isTransitioning) {
    // Global transition phase: Yellow LEDs are on for 5 seconds
    phaseDuration = 5000; // 5 seconds
    if (now - phaseStart >= phaseDuration) {
      // Transition phase complete, switch to the next command
      isTransitioning = false;
      currentCommand = nextCommand;
      nextCommand = '\0';
      currentPhase = 0; // Reset phase for the new command
      phaseStart = millis();
      allOff();
    }
    return; // Exit early, as we're in the transition phase
  }

  // Normal phase configuration
  switch(currentCommand) {
    case 'z': 
      if (currentPhase == 0) {
        // Green1 and Red2 are on
        phaseDuration = 20000; // 20 seconds
        digitalWrite(GREEN1_PIN, HIGH);
        digitalWrite(RED2_PIN, HIGH);
      } else if (currentPhase == 1) {
        // Transition: Yellow1 and Yellow2 are on for 5 seconds
        phaseDuration = 5000; // 5 seconds
        digitalWrite(Yellow1_pin, HIGH);
        digitalWrite(Yellow2_pin, HIGH);
      } else if (currentPhase == 2) {
        // Green2 and Red1 are on
        phaseDuration = 10000; // 10 seconds
        digitalWrite(GREEN2_PIN, HIGH);
        digitalWrite(RED1_PIN, HIGH);
      } else if (currentPhase == 3) {
        // Transition: Yellow1 and Yellow2 are on for 5 seconds
        phaseDuration = 5000; // 5 seconds
        digitalWrite(Yellow1_pin, HIGH);
        digitalWrite(Yellow2_pin, HIGH);
      }
      break;

    case 'x':
      if (currentPhase == 0) {
        phaseDuration = 25000; // 25 seconds
        digitalWrite(GREEN1_PIN, HIGH);
        digitalWrite(RED2_PIN, HIGH);
      } else if (currentPhase == 1) {
        phaseDuration = 5000; // 5 seconds
        digitalWrite(Yellow1_pin, HIGH);
        digitalWrite(Yellow2_pin, HIGH);
      } else if (currentPhase == 2) {
        phaseDuration = 5000; // 5 seconds
        digitalWrite(GREEN2_PIN, HIGH);
        digitalWrite(RED1_PIN, HIGH);
      } else if (currentPhase == 3) {
        phaseDuration = 5000; // 5 seconds
        digitalWrite(Yellow1_pin, HIGH);
        digitalWrite(Yellow2_pin, HIGH);
      }
      break;

    case 'c':
      if (currentPhase == 0) {
        phaseDuration = 15000; // 15 seconds
        digitalWrite(GREEN1_PIN, HIGH);
        digitalWrite(RED2_PIN, HIGH);
      } else if (currentPhase == 1) {
        phaseDuration = 5000; // 5 seconds
        digitalWrite(Yellow1_pin, HIGH);
        digitalWrite(Yellow2_pin, HIGH);
      } else if (currentPhase == 2) {
        phaseDuration = 15000; // 15 seconds
        digitalWrite(GREEN2_PIN, HIGH);
        digitalWrite(RED1_PIN, HIGH);
      } else if (currentPhase == 3) {
        phaseDuration = 5000; // 5 seconds
        digitalWrite(Yellow1_pin, HIGH);
        digitalWrite(Yellow2_pin, HIGH);
      }
      break;

    case 'v':
      if (currentPhase == 0) {
        phaseDuration = 20000; // 20 seconds
        digitalWrite(GREEN2_PIN, HIGH);
        digitalWrite(RED1_PIN, HIGH);
      } else if (currentPhase == 1) {
        phaseDuration = 5000; // 5 seconds
        digitalWrite(Yellow1_pin, HIGH);
        digitalWrite(Yellow2_pin, HIGH);
      } else if (currentPhase == 2) {
        phaseDuration = 10000; // 10 seconds
        digitalWrite(GREEN1_PIN, HIGH);
        digitalWrite(RED2_PIN, HIGH);
      } else if (currentPhase == 3) {
        phaseDuration = 5000; // 5 seconds
        digitalWrite(Yellow1_pin, HIGH);
        digitalWrite(Yellow2_pin, HIGH);
      }
      break;

    case 'b':
      if (currentPhase == 0) {
        phaseDuration = 20000; // 20 seconds
        digitalWrite(GREEN1_PIN, HIGH);
        digitalWrite(RED2_PIN, HIGH);
      } else if (currentPhase == 1) {
        phaseDuration = 5000; // 5 seconds
        digitalWrite(Yellow1_pin, HIGH);
        digitalWrite(Yellow2_pin, HIGH);
      } else if (currentPhase == 2) {
        phaseDuration = 10000; // 10 seconds
        digitalWrite(GREEN2_PIN, HIGH);
        digitalWrite(RED1_PIN, HIGH);
      } else if (currentPhase == 3) {
        phaseDuration = 5000; // 5 seconds
        digitalWrite(Yellow1_pin, HIGH);
        digitalWrite(Yellow2_pin, HIGH);
      }
      break;

    case 'n':
      if (currentPhase == 0) {
        phaseDuration = 15000; // 15 seconds
        digitalWrite(GREEN1_PIN, HIGH);
        digitalWrite(RED2_PIN, HIGH);
      } else if (currentPhase == 1) {
        phaseDuration = 5000; // 5 seconds
        digitalWrite(Yellow1_pin, HIGH);
        digitalWrite(Yellow2_pin, HIGH);
      } else if (currentPhase == 2) {
        phaseDuration = 15000; // 15 seconds
        digitalWrite(GREEN2_PIN, HIGH);
        digitalWrite(RED1_PIN, HIGH);
      } else if (currentPhase == 3) {
        phaseDuration = 5000; // 5 seconds
        digitalWrite(Yellow1_pin, HIGH);
        digitalWrite(Yellow2_pin, HIGH);
      }
      break;

    case 'm':
      if (currentPhase == 0) {
        phaseDuration = 25000; // 25 seconds
        digitalWrite(GREEN2_PIN, HIGH);
        digitalWrite(RED1_PIN, HIGH);
      } else if (currentPhase == 1) {
        phaseDuration = 5000; // 5 seconds
        digitalWrite(Yellow1_pin, HIGH);
        digitalWrite(Yellow2_pin, HIGH);
      } else if (currentPhase == 2) {
        phaseDuration = 5000; // 5 seconds
        digitalWrite(GREEN1_PIN, HIGH);
        digitalWrite(RED2_PIN, HIGH);
      } else if (currentPhase == 3) {
        phaseDuration = 5000; // 5 seconds
        digitalWrite(Yellow1_pin, HIGH);
        digitalWrite(Yellow2_pin, HIGH);
      }
      break;

    case 'a':
      if (currentPhase == 0) {
        phaseDuration = 20000; // 20 seconds
        digitalWrite(GREEN2_PIN, HIGH);
        digitalWrite(RED1_PIN, HIGH);
      } else if (currentPhase == 1) {
        phaseDuration = 5000; // 5 seconds
        digitalWrite(Yellow1_pin, HIGH);
        digitalWrite(Yellow2_pin, HIGH);
      } else if (currentPhase == 2) {
        phaseDuration = 10000; // 10 seconds
        digitalWrite(GREEN1_PIN, HIGH);
        digitalWrite(RED2_PIN, HIGH);
      } else if (currentPhase == 3) {
        phaseDuration = 5000; // 5 seconds
        digitalWrite(Yellow1_pin, HIGH);
        digitalWrite(Yellow2_pin, HIGH);
      }
      break;

    case 's':
      if (currentPhase == 0) {
        phaseDuration = 15000; // 15 seconds
        digitalWrite(GREEN1_PIN, HIGH);
        digitalWrite(RED2_PIN, HIGH);
      } else if (currentPhase == 1) {
        phaseDuration = 5000; // 5 seconds
        digitalWrite(Yellow1_pin, HIGH);
        digitalWrite(Yellow2_pin, HIGH);
      } else if (currentPhase == 2) {
        phaseDuration = 15000; // 15 seconds
        digitalWrite(GREEN2_PIN, HIGH);
        digitalWrite(RED1_PIN, HIGH);
      } else if (currentPhase == 3) {
        phaseDuration = 5000; // 5 seconds
        digitalWrite(Yellow1_pin, HIGH);
        digitalWrite(Yellow2_pin, HIGH);
      }
      break;

    default: return;
  }

  if (now - phaseStart >= phaseDuration) {
    currentPhase = (currentPhase + 1) % 4; // Loop between 0, 1, 2, and 3
    phaseStart = millis();
    allOff();
  }
}

void loop() {
  // Check for new commands first
  if (Serial.available() > 0) {
    char newCmd = Serial.read();
    if (newCmd != '\n') {
      handleCommand(newCmd);
    }
  }

  processPhase();
}
