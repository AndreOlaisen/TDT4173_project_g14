#include <zephyr.h>
#include <stdint.h>
#include <drivers/uart.h>
#include <arm_math.h>

#define IMG_DIM         32
#define IMG_CH          3
#define IMG_CATEGORIES  10

#define VCOM_UART_LABEL	DT_LABEL(DT_NODELABEL(uart1))
#define NRF_UART_LABEL	DT_LABEL(DT_NODELABEL(uart2))

int nn_forward_pass(uint8_t *img, q7_t *out);

static const struct device *vcom_uart_dev;
static const struct device *nrf_uart_dev;
static uint8_t img[IMG_DIM * IMG_DIM * IMG_CH];
static struct k_poll_signal sig_ready = K_POLL_SIGNAL_INITIALIZER(sig_ready);


static void uart_async_cb(const struct device *dev,
			  struct uart_event *evt,
			  void *user_data)
{
	if (dev == vcom_uart_dev && evt->type == UART_RX_DISABLED) {
		k_poll_signal_raise(&sig_ready, 0);
	}
}

static void uart_init(const struct device **dev, const char *label)
{
	*dev = device_get_binding(label);
	__ASSERT(*dev != NULL, "Unable to get UART device.");
	int err = uart_callback_set(*dev, uart_async_cb, NULL);
	__ASSERT(err == 0, "Error %d setting UART callback.", err);
}


void main(void)
{
	q7_t out[IMG_CATEGORIES];

	printk("Starting NN example.\n");
	
	k_poll_signal_init(&sig_ready);
	struct k_poll_event evt_ready = K_POLL_EVENT_INITIALIZER(
		K_POLL_TYPE_SIGNAL, K_POLL_MODE_NOTIFY_ONLY, &sig_ready);

	uart_init(&vcom_uart_dev, VCOM_UART_LABEL);
	uart_init(&nrf_uart_dev, NRF_UART_LABEL);

	for (;;) {
		uart_rx_enable(vcom_uart_dev, img, sizeof(img), SYS_FOREVER_MS);
		k_poll(&evt_ready, 1, K_FOREVER);
		evt_ready.signal->signaled = 0;
		evt_ready.state = K_POLL_STATE_NOT_READY;
		printk("Image received - performing forward pass\n");

		nn_forward_pass(img, out);

		uart_tx(vcom_uart_dev, (uint8_t *) out, sizeof(out),
			SYS_FOREVER_MS);
		uart_tx(nrf_uart_dev, (uint8_t *) out, sizeof(out),
			SYS_FOREVER_MS);
		printk("Sent output:\n");
		for (int i = 0; i < IMG_CATEGORIES; i++) {
			printk("    %d: %hhd\n", i, out[i]);
		}
	}
}
